"""Written by David McDougall, 2018"""

import numpy as np
from sdr import SDR
from synapses import SynapseManager
  
# TODO: Make a parameter "diagnostics_alpha" for measuring mini-column
# activation frequency and average overlap insead of using "boosting_alpha".

class SpatialPooler:
    """
    This class handles the mini-column structures and the feed forward 
    proximal inputs to each cortical mini-column.

    [CITE THE SP PAPER HERE]

    Topology: This implements local inhibition with topology by creating many
    small groups of mini-columns which are distributed across the input space.
    All of the mini-columns in a group are located at the same location in the
    input space, and they inhibit each other equally.   Each group of mini-
    columns is self contained; groups of mini-columns do not inhibit each other
    or interact.  Instead of creating a large spatial pooler with topology, this
    creates many small spatial poolers with topology between the spatial
    poolers.
    """
    def __init__(self,
        input_sdr,
        mini_columns,     # Integer,
        sparsity,
        potential_pool,
        permanence_inc,
        permanence_dec,
        permanence_thresh,
        segments            = 1,
        macro_columns       = (1,),
        init_dist           = (0, 0),
        boosting_alpha      = None,
        active_thresh       = 0,
        radii               = tuple()):
        """
        Argument mini_columns is an Integer, the number of mini-columns in each 
            macro-column.

        Argument macro_columns is a tuple of integers.  Dimensions of macro
            column array.  These are topological dimensions.  Macro columns are
            distributed across the input space in a uniform grid.

        Optional Argument radii defines the input topology.  Trailing extra
            input dimensions are treated as non-topological dimensions.

        Argument segments is an Integer, number proximal segments for each
            mini-column.

        Argument sparsity ...

        Argument potential_pool ...

        Optional Argument boosting_alpha is the small constant used by the
        moving exponential average which tracks each mini-columns activation
        frequency.  Default value is None, which disables boosting altogether.

        Argument permanence_inc ...
        Argument permanence_dec ...
        Argument permanence_thresh ...
        Argument init_dist is (mean, std) of initial permanence values, which is a
                 gaussian random distribution.

        Argument active_thresh ...
        """
        assert(isinstance(input_sdr, SDR))
        assert(potential_pool > 1) # Number of synapses, not percent.
        self.mini_columns     = int(round(mini_columns))
        self.macro_columns    = tuple(int(round(dim)) for dim in macro_columns)
        self.radii            = radii
        self.segments         = int(round(segments))
        self.columns          = SDR(self.macro_columns + (self.mini_columns,),
            activation_frequency_alpha = boosting_alpha,
            average_overlap_alpha      = boosting_alpha,)
        self.sparsity         = sparsity
        self.active_thresh    = active_thresh
        self.potential_pool   = potential_pool
        self.age              = 0

        segment_shape = self.macro_columns + (self.mini_columns, self.segments)
        self.synapses = SynapseManager(
            input_sdr              = input_sdr,
            output_sdr             = SDR(segment_shape),
            radii                  = radii,
            init_dist              = init_dist,
            permanence_inc         = permanence_inc,
            permanence_dec         = permanence_dec,
            permanence_thresh      = permanence_thresh,
            initial_potential_pool = self.potential_pool,)

        if True:
            # Nupic's SP init method
            # TODO: Make this a permanent part of the synapses class?  
            # Change init_dist argument to accept a string 'sp' ?
            for idx in range(self.synapses.output_sdr.size):
                pp = self.synapses.postsynaptic_permanences[idx].shape[0]
                connnected  = np.random.random(pp) > .5
                permanences = np.random.random(pp)
                permanences[connnected] *= 1 - self.synapses.permanence_thresh
                permanences[connnected] += self.synapses.permanence_thresh
                permanences[np.logical_not(connnected)] *= self.synapses.permanence_thresh
                self.synapses.postsynaptic_permanences[idx] = np.array(permanences, dtype=np.float32)
            self.synapses.rebuild_indexes()

        # Break ties randomly, in a constant unchanging manner.
        self.tie_breakers = np.random.uniform(0, .5, size=self.synapses.output_sdr.dimensions)

        self.boosting_alpha = boosting_alpha
        if boosting_alpha is not None:
            # Make a dedicated SDR to track segment activation frequencies for
            # boosting.
            self.boosting = SDR(self.synapses.output_sdr,
                                activation_frequency_alpha = boosting_alpha,
                                average_overlap_alpha      = boosting_alpha,)
            # Initialize to the target activation frequency/sparsity.
            self.boosting.activation_frequency.fill(self.sparsity / self.segments)

        self.reset()

    def reset(self):
        self.columns.zero()
        self.prev_updates = np.full(self.synapses.output_sdr.size, None)

    def compute(self, input_sdr=None, input_learning_sdr=None, learn=True):
        """
        """
        excitement, potential_excitement = self.synapses.compute(input_sdr=input_sdr)
        excitement = excitement + self.tie_breakers

        # Logarithmic Boosting Function.
        if self.boosting_alpha is not None:
            target_sparsity = self.sparsity / self.segments
            boost = np.log2(self.boosting.activation_frequency) / np.log2(target_sparsity)
            boost = np.nan_to_num(boost)
            excitement *= boost

        # Divide excitement by the number of connected synapses.
        n_con_syns = self.synapses.postsynaptic_connected_count
        n_con_syns = n_con_syns.reshape(self.synapses.output_sdr.dimensions)
        percent_overlap = excitement / n_con_syns

        # Reduce the segment dimension to each mini-columns single most excited
        # segment.
        column_excitement = np.max(percent_overlap, axis=-1)

        # Stable SP and Grid Cells modify the excitement here.
        column_excitement = self._compute_hook(column_excitement)

        # Activate mini-columns.  First determine how many mini-columns to
        # activate in each macro-column.
        n_activate = max(1, int(round(self.mini_columns * self.sparsity)))

        # Activate the most excited mini-columns in each macro-column.
        k = self.mini_columns - n_activate
        mini_index = np.argpartition(column_excitement, k, axis=-1)[..., k:]

        # Convert activations from mini-column indices to macro-column indices.
        macro_index    = tuple(np.indices(mini_index.shape))[:-1]
        winner_columns = tuple(x.reshape(-1) for x in macro_index + (mini_index,))
        # Filter out columns with sub-threshold excitement.
        winner_excitement = np.max(excitement[winner_columns], axis=-1)
        winner_columns    = tuple(np.compress(winner_excitement >= self.active_thresh,
                                      winner_columns, axis=1))

        # Output the results into the columns sdr.
        self.columns.index = winner_columns

        if learn:
            seg_idx = np.argmax(excitement[winner_columns], axis=-1)
            learning_segments = winner_columns + (seg_idx,)
            self.prev_updates = self.synapses.learn(
                input_sdr    = input_learning_sdr,
                output_sdr   = learning_segments,
                prev_updates = self.prev_updates,)

            # Update the exponential moving average of each segments activation frequency.
            if self.boosting_alpha is not None:
                self.boosting.assign(learning_segments)

            self.age += 1

        return self.columns

    def _compute_hook(self, x):
        """Subclasses override this method."""
        return x

    def statistics(self, _class_name='Spatial Pooler'):
        stats = _class_name + ' '
        stats += self.synapses.statistics()

        stats += 'Columns ' + self.columns.statistics()
        if self.boosting_alpha is not None and self.segments > 1:
            stats      += 'Segments ' + self.boosting.statistics()
            af         = self.boosting.activation_frequency
            target     = self.sparsity / self.segments
            boost_min  = np.log2(np.min(af))  / np.log2(target)
            boost_mean = np.log2(np.mean(af)) / np.log2(target)
            boost_max  = np.log2(np.max(af))  / np.log2(target)
            stats += '\tLogarithmic Boosting Multiplier min/mean/max  {:-.04g}% / {:-.04g}% / {:-.04g}%\n'.format(
                    boost_min   * 100,
                    boost_mean  * 100,
                    boost_max   * 100,)
        return stats


class StableSpatialPooler(SpatialPooler):
    def __init__(self, stability_rate, **kw_args):
        SpatialPooler.__init__(self, **kw_args)
        assert(stability_rate >= 0 and stability_rate <= 1)
        self.stability_rate = stability_rate

    def reset(self):
        SpatialPooler.reset(self)
        self.X_act = np.zeros(self.columns.dimensions)

    def _compute_hook(self, excitement):
        self.X_act += self.stability_rate * (excitement - self.X_act)
        return self.X_act

    def statistics(self):
        return SpatialPooler.statistics(self, _class_name='Stable Spatial Pooler')


class GridCells(SpatialPooler):
    def __init__(self, stability_rate, fatigue_rate, **kw_args):
        self.stability_rate = stability_rate
        self.fatigue_rate   = fatigue_rate
        SpatialPooler.__init__(self, **kw_args)

    def reset(self):
        self.X_act   = np.zeros(self.columns.dimensions)
        self.X_inact = np.zeros(self.columns.dimensions)
        SpatialPooler.reset(self)

    def _compute_hook(self, excitement):
        self.X_act   += self.stability_rate * (excitement - self.X_act - self.X_inact)
        self.X_inact += self.fatigue_rate   * (excitement - self.X_inact)
        return self.X_act

    def statistics(self):
        return SpatialPooler.statistics(self, _class_name='Grid Cells')
