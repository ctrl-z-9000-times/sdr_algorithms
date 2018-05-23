# Written by David McDougall, 2018

import numpy as np
import random
from sdr import SDR
from synapses import SynapseManager

# TODO: This should break ties in a stable way, not with a new set of random tie
# breakers every cycle...

# TODO: This should record the natural-stability, which is the stability of the
# SP without the min-stab mechanism.  Measure it inside of the stabilize method.

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
    or interact.
    """
    def __init__(self,
        mini_columns,     # Integer,
        segments,         # Integer, proximal segments per minicolumns
        sparsity,
        potential_pool,
        permanence_inc,
        permanence_dec,
        permanence_thresh,
        init_dist,
        input_sdr,
        macro_columns,
        boosting_alpha = None,
        active_thresh = 0,
        min_stability = None,
        radii=tuple()):
        """
        Argument parameters is an instance of SpatialPoolerParameters.

        Argument mini_columns ...

        Argument macro_columns is a tuple of integers.  Dimensions of macro
                 column array.

        Optional Argument radii defines the input topology.  Trailing extra
            input dimensions are treated as non-topological dimensions.

        Argument segments is the number of coincidence detectors to make for each
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

        Argument min_stability ... set to None to disable.
        """
        assert(isinstance(input_sdr, SDR))
        self.mini_columns   = int(round(mini_columns))
        self.macro_columns  = tuple(int(round(dim)) for dim in macro_columns)
        self.segments       = int(round(segments))
        self.columns        = SDR(self.macro_columns + (self.mini_columns,))
        self.sparsity       = sparsity
        self.active_thresh  = active_thresh
        self.age            = 0

        segment_shape = self.macro_columns + (self.mini_columns, self.segments)
        self.synapses = SynapseManager( input_sdr,
                                        SDR(segment_shape),
                                        radii             = radii,
                                        init_dist         = init_dist,
                                        permanence_inc    = permanence_inc,
                                        permanence_dec    = permanence_dec,
                                        permanence_thresh = permanence_thresh,)

        self.synapses.normally_distributed_connections(potential_pool, radii)

        self.boosting_alpha = boosting_alpha
        if boosting_alpha is not None:
            # Make a dedicated SDR to track segment activation frequencies for
            # boosting.
            self.boosting = SDR(self.synapses.output_sdr,
                                activation_frequency_alpha = boosting_alpha)
            # Initialize to the target activation frequency/sparsity.
            self.boosting.activation_frequency.fill(sparsity / self.segments)

        self.min_stability = min_stability
        if min_stability is not None:
            self.prev_columns = SDR(self.columns)

    def reset(self):
        if self.min_stability is not None:
            self.prev_columns.zero()

    def compute(self, input_sdr=None):
        """
        """
        excitement, potential_excitement = self.synapses.compute(input_sdr=input_sdr)

        # Logarithmic Boosting Function.
        if self.boosting_alpha is not None:
            target_sparsity = self.sparsity / self.segments
            boost = np.log2(self.boosting.activation_frequency) / np.log2(target_sparsity)
            boost = np.nan_to_num(boost).reshape(self.boosting.dimensions)
            excitement = boost * excitement

        # Break ties randomly.
        excitement = excitement + np.random.uniform(0, .5, size=self.synapses.output_sdr.dimensions)

        # Reduce the segment dimension to each mini-columns single most excited
        # segment.
        self.segment_excitement = excitement     # Needed for learning.
        column_excitement = np.max(excitement, axis=-1)

        # Activate the most excited mini-columns in each macro-column.
        k = self.mini_columns * self.sparsity
        k = max(1, int(round(k)))
        mini_index  = np.argpartition(-column_excitement, k-1, axis=-1)[..., :k]

        # Convert activations from mini-column indices to macro-column indices.
        macro_index = tuple(np.indices(mini_index.shape))[:-1] + (mini_index,)
        selected_columns = tuple(vec.reshape(-1) for vec in macro_index)
        # Filter out columns with sub-threshold excitement.
        selected_excitement = column_excitement[selected_columns]
        selected_columns    = tuple(np.compress(selected_excitement >= self.active_thresh,
                                          selected_columns, axis=1))
        self.columns.index  = tuple(selected_columns)

        if self.min_stability is not None:
            self.stabilize(column_excitement)

        return self.columns

    def stabilize(self, column_excitement):
        """
        This activates prior columns to force active in order to maintain
        (self.min_stability) percent of column overlap between time steps.  Always call
        this between compute and learn!

        Argument column_excitement is array of each mini-column's excitement.
        """
        num_active      = len(self.columns)
        overlap         = self.columns.overlap(self.prev_columns)
        stabile_columns = int(round(num_active * overlap))
        target_columns  = int(round(num_active * self.min_stability))
        add_columns     = target_columns - stabile_columns
        if add_columns <= 0:
            return

        eligable_columns  = np.setdiff1d(self.prev_columns.flat_index, self.columns.flat_index)
        eligable_excite   = column_excitement.reshape(-1)[eligable_columns]
        add_columns       = min(add_columns, len(eligable_excite))
        selected_col_nums = np.argpartition(-eligable_excite, add_columns-1)[:add_columns]
        selected_columns  = eligable_columns[selected_col_nums]
        selected_index    = np.unravel_index(selected_columns, self.columns.dimensions)
        self.columns.flat_index = np.concatenate([self.columns.flat_index, selected_columns])
        self.prev_columns.assign(self.columns)

    def learn(self, column_sdr=None):
        """
        Make the spatial pooler learn about its current inputs and active columns.
        """
        self.columns.assign(column_sdr)
        seg_idx = np.argmax(self.segment_excitement[self.columns.index], axis=-1)
        learning_segments = self.columns.index + (seg_idx,)
        self.synapses.learn(output_sdr=learning_segments)
        # Update the exponential moving average of each segments activation frequency.
        if self.boosting_alpha is not None:
            self.boosting.assign(self.synapses.output_sdr)
        self.age += 1

    def statistics(self):
        stats = 'Spatial Pooler '
        stats += self.synapses.statistics()

        if self.boosting_alpha is not None:
            stats      += 'Columns ' + self.boosting.statistics()
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
