# Written by David McDougall, 2018

import numpy as np
import random
from sdr import SDR
from synapses import SynapseManager

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
        mini_columns,     # Integer,
        segments,         # Integer, proximal segments per minicolumns
        sparsity,
        potential_pool,
        permanence_inc,
        permanence_dec,
        permanence_thresh,
        input_sdr,
        macro_columns,
        init_dist        = (0, 0),
        max_add_synapses = 0,
        boosting_alpha = None,
        active_thresh = 0,
        min_stability = None,
        radii=tuple()):
        """
        Argument parameters is an instance of SpatialPoolerParameters.

        Argument mini_columns is the number of mini-columns in each 
            macro-column.

        Argument macro_columns is a tuple of integers.  Dimensions of macro
            column array.  These are topological dimensions.  Macro columns are
            distributed across the input space in a uniform grid.

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
        self.radii          = radii
        self.potential_pool = potential_pool
        self.max_add_synapses = max_add_synapses
        self.age            = 0

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
                                activation_frequency_alpha = boosting_alpha)
            # Initialize to the target activation frequency/sparsity.
            self.boosting.activation_frequency.fill(self.sparsity / self.segments)

        self.min_stability = min_stability
        if min_stability is not None:
            self.prev_columns = SDR(self.columns)

    def reset(self):
        self.columns.zero()
        if self.min_stability is not None:
            self.prev_columns.zero()

    def compute(self, input_sdr=None):
        """
        """
        excitement, potential_excitement = self.synapses.compute(input_sdr=input_sdr)
        self.potential_excitement = potential_excitement

        excitement = excitement + self.tie_breakers

        # Logarithmic Boosting Function.
        if self.boosting_alpha is not None:
            target_sparsity = self.sparsity / self.segments
            boost = np.log2(self.boosting.activation_frequency) / np.log2(target_sparsity)
            boost = np.nan_to_num(boost).reshape(self.boosting.dimensions)
            excitement = boost * excitement

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

        # if learn:
        #     self.learn()

        return self.columns

    def stabilize(self,
        # active_columns,
        # prev_active_columns,
        column_excitement,
        learn=True):
        """
        This activates prior mini-columns to force active in order to maintain
        (self.min_stability) percent of mini-column overlap between time steps.
        Always call this between compute and learn!

        Argument active_columns is a tuple of index arrays
        Argument prev_active_columns is an SDR
        Argument column_excitement is array of each mini-column's excitement.

        Returns index tuple of stablizing column activations.
        """
        # Determine how many previously active mini-columns are still active in
        # each macro-column.
        stable_columns    = np.logical_and(self.prev_columns.dense, self.columns.dense)
        natural_stability = np.sum(stable_columns, axis = -1)
        num_prev_columns  = np.sum(self.prev_columns.dense, axis = -1)
        target_stability  = int(round(self.mini_columns * self.sparsity * self.min_stability))
        # add_columns is the number of mini-columns to activate in each macro-
        # column in order to meet the target stability.
        add_columns       = target_stability - natural_stability
        add_columns       = np.maximum(0, add_columns).reshape(-1)

        # Find the average natural stability for the whole SP.
        target_active          = round(self.columns.size * self.sparsity)
        self.natural_stability = np.sum(natural_stability) / target_active

        # Search for minicolumns which were previously active and are no longer
        # active.
        eligable_columns = np.logical_and(self.prev_columns.dense, np.logical_not(self.columns.dense))
        # Apply min-stability to each macro column.
        eligable_columns  = eligable_columns.reshape(-1, self.mini_columns)
        column_excitement = column_excitement.reshape(-1, self.mini_columns)
        n_macrocolumns    = np.product(self.macro_columns)
        # selected_columns accumulates mini-columns to activate.
        # selected_columns is a pair of index arrays: (macro-column-index, mini-
        # column-index).
        selected_columns  = ([], [])
        for macro_index in range(n_macrocolumns):
            # Find the indices of eligable mini-columns in this macro-column.
            eligable_mini = np.nonzero(eligable_columns[macro_index])[0]
            n_add_cols    = min(add_columns[macro_index], len(eligable_mini))
            if n_add_cols == 0:
                continue
            # Gather the excitement of eligable mini-columns.
            eligable_excite = column_excitement[macro_index, eligable_mini]
            # Find the most excited eligable mini-columns in this macro-column.
            selected_nums   = np.argpartition(-eligable_excite, n_add_cols-1)[:n_add_cols]
            selected_mini   = eligable_mini[selected_nums]
            selected_columns[0].extend([macro_index] * n_add_cols)
            selected_columns[1].extend(selected_mini)
        # Determine the flat index for the selected mini-columns.  Then add them
        # to the active mini-columns SDR.
        if selected_columns[0]:     # Empty multi_index crashes np.ravel_multi_index
            selected_flat_idx = np.ravel_multi_index(
                multi_index = selected_columns,
                dims        = (n_macrocolumns, self.mini_columns))
            self.columns.flat_index = np.concatenate([self.columns.flat_index, selected_flat_idx])
        # Cycle self.prev_columns in preparation for the next cycle.
        self.prev_columns.assign(self.columns)

    def learn(self, column_sdr=None):
        """
        Make the spatial pooler learn about its current inputs and active columns.
        """
        self.columns.assign(column_sdr)
        seg_idx = np.argmax(self.segment_excitement[self.columns.index], axis=-1)
        learning_segments = self.columns.index + (seg_idx,)
        self.synapses.learn(output_sdr=learning_segments)

        if self.max_add_synapses > 0:
            time_to_zero = self.synapses.permanence_thresh / self.synapses.permanence_dec
            prune_period = int(100 * time_to_zero)
            if self.age % prune_period == 0:
                # print("SP.remove_zero_permanence_synapses")
                self.synapses.remove_zero_permanence_synapses()

            # Add synapses to active segments until they have
            # self.max_add_synapses many active presynapses.
            # First sort the active outputs B/C add_synapses requires it.
            self.synapses.output_sdr.flat_index = np.sort(self.synapses.output_sdr.flat_index)
            potential_excitement = self.potential_excitement[self.synapses.output_sdr.index]
            potential_excitement.dtype = np.int32   # Otherwise this will add 4 billion new synapses.
            new_synapse_count = np.maximum(0, self.max_add_synapses - potential_excitement)
            # self.synapses.output_sdr.nz_values = np.array(new_synapse_count, dtype=np.uint8)
            self.synapses.add_synapses(
                num_synapses         = new_synapse_count,
                maximum_new_synapses = self.max_add_synapses,
                maximum_synapses     = self.potential_pool,
                init_dist            = (0, 0),)

        # Update the exponential moving average of each segments activation frequency.
        if self.boosting_alpha is not None:
            self.boosting.assign(learning_segments)

        self.age += 1

    def statistics(self):
        stats = 'Spatial Pooler '
        stats += self.synapses.statistics()

        if self.boosting_alpha is not None:
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
