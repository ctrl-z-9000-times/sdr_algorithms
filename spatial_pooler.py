# Written by David McDougall, 2018

import numpy as np
from sdr import SDR
from synapses import SynapseManager
  
# TODO: Make a parameter "diagnostics_alpha" for measuring mini-column
# activation frequency and average overlap insead of using "boosting_alpha".
#
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
        sparsity,
        potential_pool,
        permanence_inc,
        permanence_dec,
        permanence_thresh,
        input_sdr,
        segments            = 1,         # Integer, proximal segments per minicolumns
        macro_columns       = (1,),
        init_dist           = (0, 0),
        boosting_alpha      = None,
        active_thresh       = 0,
        min_stability       = None,
        max_add_synapses    = None,    # Fraction of synapses needed for activation to add. Range [0, 1]
        min_add_synapses    = None,    # Target excitement level, add at least this many synapses. Integer
        matching_thresh     = None,
        radii               = tuple()):
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
        self.mini_columns     = int(round(mini_columns))
        self.macro_columns    = tuple(int(round(dim)) for dim in macro_columns)
        self.segments         = int(round(segments))
        self.min_stability    = min_stability
        self.columns          = SDR(self.macro_columns + (self.mini_columns,),
            activation_frequency_alpha = boosting_alpha,
            average_overlap_alpha      = boosting_alpha,)
        self.sparsity         = sparsity
        self.active_thresh    = active_thresh
        self.matching_thresh  = matching_thresh
        self.radii            = radii
        self.potential_pool   = potential_pool
        assert(potential_pool > 1) # Number of synapses, not percent.
        self.max_add_synapses = max_add_synapses
        self.min_add_synapses = min_add_synapses
        self.age              = 0

        init_pp = self.potential_pool
        if self.min_stability is not None:
            # If SP is going to be adding synapses then leave room to grow.
            init_pp = int(init_pp / 2)

        segment_shape = self.macro_columns + (self.mini_columns, self.segments)
        self.synapses = SynapseManager(
            input_sdr              = input_sdr,
            output_sdr             = SDR(segment_shape),
            radii                  = radii,
            init_dist              = init_dist,
            permanence_inc         = permanence_inc,
            permanence_dec         = permanence_dec,
            permanence_thresh      = permanence_thresh,
            initial_potential_pool = init_pp,)

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

        if min_stability is not None:
            self.prev_columns = SDR(self.columns)

    def reset(self):
        self.columns.zero()
        if self.min_stability is not None:
            self.prev_columns.zero()

    def compute(self, input_sdr=None, input_learning_sdr=None, learn=True):
        """
        Attribute natural_stability, is always set.  Is the fraction of
            previously active mini-columns which are winner columns this cycle.
        """
        excitement, potential_excitement = self.synapses.compute(input_sdr=input_sdr)
        excitement = excitement + self.tie_breakers

        # Logarithmic Boosting Function.
        if self.boosting_alpha is not None:
            target_sparsity = self.sparsity / self.segments
            boost = np.log2(self.boosting.activation_frequency) / np.log2(target_sparsity)
            boost = np.nan_to_num(boost)
            excitement = boost * excitement

        # Reduce the segment dimension to each mini-columns single most excited
        # segment.
        column_excitement = np.max(excitement, axis=-1)

        # Activate mini-columns.  First determine how many mini-columns to
        # activate in each macro-column.
        n_activate = max(1, int(round(self.mini_columns * self.sparsity)))

        # Take care of minimum stability columns first so that this keeps a
        # constant sparsity.
        if self.min_stability is not None:
            # Determine the minimum excitement to win in each macro-columns,
            # assuming that minimum stability is zero (natural stability).
            k = self.mini_columns - n_activate
            min_winner = np.argpartition(column_excitement, k, axis=-1)[..., k]
            min_winner = tuple(np.indices(min_winner.shape)) + (min_winner,)
            min_winning_excitement = column_excitement[min_winner]

            n_stabilize = int(round(n_activate * self.min_stability))

            stabilized_columns = self._stabilize(
                self.prev_columns,
                column_excitement,
                n_stabilize,)

            # Determine the natural stability for diagnostics.
            stabilized_excitement = column_excitement[stabilized_columns]
            excitement_threshold = min_winning_excitement[stabilized_columns[ : -1]]
            nat_stab = np.count_nonzero(stabilized_excitement >= excitement_threshold)
            self.natural_stability = nat_stab / (n_activate * np.product(self.macro_columns))

            # Prevent stabilized columns from participating in the competition
            # and activating second time.
            column_excitement[stabilized_columns] = self.active_thresh - 1
            n_activate = max(1, n_activate - n_stabilize)

        winner_columns = self._winners(column_excitement, n_activate)

        # Put the results together and assign them into the output columns sdr.
        if self.min_stability is not None:
            self.columns.index = np.concatenate([winner_columns, stabilized_columns], axis=1)
            # Cycle self.prev_columns in preparation for the next cycle.
            self.columns.dense  # Make this exist before the assignment tries to copy it.
            self.prev_columns.assign(self.columns)
        else:
            self.columns.index = winner_columns

        if learn:
            self.synapses.input_sdr.assign(input_learning_sdr)

            self._learn_winners(winner_columns, excitement)

            if self.min_stability is not None:
                self._learn_stabilized(
                    stabilized_columns,
                    potential_excitement,
                    min_winning_excitement)

            self.age += 1

        return self.columns

    def _winners(self, column_excitement, n_activate):
        """
        Returns index tuple of winner columns
        """
        # Activate the most excited mini-columns in each macro-column.
        k = self.mini_columns - n_activate
        mini_index = np.argpartition(column_excitement, k, axis=-1)[..., k:]

        # Convert activations from mini-column indices to macro-column indices.
        macro_index    = tuple(np.indices(mini_index.shape))[:-1]
        winner_columns = tuple(x.reshape(-1) for x in macro_index + (mini_index,))
        # Filter out columns with sub-threshold excitement.
        winner_excitement = column_excitement[winner_columns]
        winner_columns    = tuple(np.compress(winner_excitement >= self.active_thresh,
                                      winner_columns, axis=1))
        return winner_columns

    def _learn_winners(self, columns_index, segment_excitement):
        """
        Make the spatial pooler learn about its current inputs and winner columns.
        """
        seg_idx = np.argmax(segment_excitement[columns_index], axis=-1)
        learning_segments = columns_index + (seg_idx,)
        self.synapses.learn(output_sdr=learning_segments)

        # Update the exponential moving average of each segments activation frequency.
        if self.boosting_alpha is not None:
            self.boosting.assign(learning_segments)

    def _stabilize(self,
        prev_active_columns,
        column_excitement,
        n_stabilize):
        """
        Activate mini-columns which were active in the previous cycle in order
        to maintain at least (self.min_stability) percent of mini-column overlap
        between time steps.

        Argument prev_active_columns is an SDR
        Argument column_excitement is array of each mini-column's excitement.

        Returns index tuple of stablizing column activations.
        """
        # Search for mini-columns which were previously active in each macro-
        # column.  Select the currently most excited mini-columns to activate.
        n_macrocolumns    = np.product(self.macro_columns)
        eligable_columns  = prev_active_columns.dense.reshape(n_macrocolumns, self.mini_columns)
        column_excitement = column_excitement.reshape(n_macrocolumns, self.mini_columns)
        # stabilized_macro_columns and stabilized_mini_columns accumulate mini-
        # column indices to activate.  The macro columns list contains a
        # flattened index which is unraveled at the end.
        stabilized_macro_columns = []
        stabilized_mini_columns  = []
        for macro_index in range(n_macrocolumns):
            # Find the indices of eligable mini-columns in this macro-column.
            eligable_mini = np.nonzero(eligable_columns[macro_index])[0]
            n_add_cols    = min(n_stabilize, len(eligable_mini))
            if n_add_cols == 0:
                continue
            # Gather the excitement of eligable mini-columns.
            eligable_excite = column_excitement[macro_index, eligable_mini]
            # Find the most excited eligable mini-columns in this macro-column.
            # selected_nums = np.argpartition(-eligable_excite, n_add_cols-1)[:n_add_cols]
            k = len(eligable_excite) - n_add_cols
            selected_nums = np.argpartition(eligable_excite, k)[k :]
            selected_mini = eligable_mini[selected_nums]
            stabilized_macro_columns.extend([macro_index] * n_add_cols)
            stabilized_mini_columns.extend(selected_mini)
        # Convert the macro-column index to the correct dimensions.
        stabilized_macro_columns = np.array(stabilized_macro_columns, dtype=np.int)
        stabilized_macro_columns = np.unravel_index(stabilized_macro_columns, self.macro_columns)
        stabilized_mini_columns  = np.array(stabilized_mini_columns, dtype=np.int)
        return stabilized_macro_columns + (stabilized_mini_columns,)

    def _learn_stabilized(self,
        stabilized_columns,
        potential_excitement,
        min_winning_excitement):
        """
        Make the Spatial Pooler learn about its current inputs and the
        stabilized mini-columns.  This initializes new segments and adds
        synapses as needed.
        """
        # Prune out dead / zero permanence synapses.
        if bool(self.max_add_synapses): # Only remove synapses if it can add more later.
            time_to_zero = self.synapses.permanence_thresh / self.synapses.permanence_dec
            prune_period = int(10 * time_to_zero)
            if self.age % prune_period == prune_period - 1:
                self.synapses.remove_zero_permanence_synapses()

        # Search for the best matching segment in each stabilized mini-column to
        # learn.  
        potential_dense       = potential_excitement[stabilized_columns]
        best_matching_segment = np.argmax(potential_dense, axis = -1)
        matching_potential    = potential_dense[np.arange(best_matching_segment.shape[0]), best_matching_segment]

        # Apply a threshold to be rid of weak and insiginificant matches.  The
        # threshold is a fraction of the excitement which would have been needed
        # to activate.
        winning_excitement   = min_winning_excitement[stabilized_columns[ : -1]]
        winning_excitement   = np.ceil(winning_excitement) # When in doubt, assume the competition is stronger rather than weaker.
        matching_threshold   = self.matching_thresh * winning_excitement
        subthreshold_matches = matching_potential < matching_threshold

        # If no good segments exist then start new segments on the least used
        # segment, by number of potential synapses.
        needs_new_segment  = tuple(idx[subthreshold_matches] for idx in stabilized_columns)
        potential_pools    = self.synapses.postsynaptic_sources.reshape(self.synapses.output_sdr.dimensions)
        potential_pools    = potential_pools[needs_new_segment]
        array_size         = lambda pp: pp.shape[0]
        potential_pools_sz = np.frompyfunc(array_size, 1, 1)(potential_pools)
        new_segments       = np.argmin(potential_pools_sz, axis = -1)

        # Put together the best matching and the newly created segments into an
        # index of learning segments.
        best_matching_segment[subthreshold_matches] = new_segments
        learning_segments = stabilized_columns + (best_matching_segment,)

        # Sort the learning_segments/output_sdr B/C add_synapses requires it.
        self.synapses.output_sdr.index = learning_segments  # Assign to index, access flat_index, conversion done by SDR class.
        sort_order = np.argsort(self.synapses.output_sdr.flat_index)
        self.synapses.output_sdr.flat_index = self.synapses.output_sdr.flat_index[sort_order]
        # Add synapses to learning segments until they have enough active
        # presynapses to activate by winning the competition.
        potential_excitement = potential_excitement[self.synapses.output_sdr.index]
        potential_excitement.dtype = np.int32   # Otherwise this may add 4 billion new synapses.
        winning_excitement = winning_excitement[sort_order]
        winning_excitement = np.maximum(self.min_add_synapses, winning_excitement) # Require at enough synapses to activate.
        new_synapse_count = (winning_excitement - potential_excitement)
        new_synapse_count = np.maximum(0, self.max_add_synapses * new_synapse_count)
        new_synapse_count = np.array(np.rint(new_synapse_count), dtype=np.int)
        if len(self.synapses.output_sdr) > 0:
            real_max_add_syn = np.max(new_synapse_count)
            self.synapses.add_synapses(
                num_synapses         = new_synapse_count,
                maximum_new_synapses = real_max_add_syn,
                maximum_synapses     = self.potential_pool,
                init_dist            = (0, 0),)

        self.synapses.learn()

    def statistics(self):
        stats = 'Spatial Pooler '
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
