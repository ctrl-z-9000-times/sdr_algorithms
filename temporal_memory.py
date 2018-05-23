# Written by David McDougall, 2018

import numpy as np
import random
from sdr import SDR
from synapses import SynapseManager


class TemporalMemory:
    """
    This implementation is based on the paper: Hawkins J. and Ahmad S. (2016)
    Why Neurons Have Thousands of Synapses, a Theory of Sequency Memory in
    Neocortex. Frontiers in Neural Circuits 10:23 doi: 10.3389/fncir.2016.00023
    """
    def __init__(self, 
        column_sdr,
        add_synapses,             # Add more synapses segments until they have this many active presynapses.
        cells_per_column,
        init_dist,                # Initial distribution of permanence values.
        segments_per_cell,
        synapses_per_segment,
        permanence_inc,
        permanence_dec,
        mispredict_dec,
        permanence_thresh,
        predictive_threshold,     # Segment excitement threshold for predictions.
        learning_threshold,       # Segment excitement threshold for learning.
        predicted_boost,          # Predicted cells activate this many times.
        context_sdr=None,
        anomaly_alpha = 1/1000,
        ):
        """
        Argument parameters is an instance of TemporalMemoryParameters
        Argument column_dimensions ...

        Attribute active is SDR, shape = (num_columns, cells_per_column)

        Attribute learning is SDR, shape = (num_columns, cells_per_column)

        """
        assert(isinstance(column_sdr,  SDR))
        assert(isinstance(context_sdr, SDR) or context_sdr is None)
        self.columns             = column_sdr
        self.context_sdr         = context_sdr
        self.cells_per_column    = int(round(cells_per_column))
        if self.cells_per_column < 1:
            raise ValueError("Cannot create TemporalMemory with cells_per_column < 1.")
        self.segments_per_cell   = int(round(segments_per_cell))
        assert(self.segments_per_cell > 0)
        self.active              = SDR((self.columns.size, self.cells_per_column),
                                        activation_frequency_alpha = 1/1000,)
        self.learning            = SDR(self.active)
        self.anomaly_alpha       = anomaly_alpha
        self.mean_anomaly        = 1.0

        if context_sdr is None:
            context_sdr = self.active

        self.synapses_per_segment = int(round(synapses_per_segment))
        self.add_synapses         = int(round(add_synapses))
        self.predictive_threshold = predictive_threshold
        self.learning_threshold   = learning_threshold
        self.mispredict_dec       = mispredict_dec
        if self.learning_threshold > self.predictive_threshold:
            raise ValueError("Learning Threshold excedes Predictive Threshold.")
        self.predicted_boost      = predicted_boost

        self.synapses = SynapseManager(
            input_sdr         = SDR(context_sdr),
            output_sdr        = SDR((self.columns.size, self.cells_per_column, self.segments_per_cell)),
            radii             = None,
            init_dist         = init_dist,
            permanence_thresh = permanence_thresh,
            permanence_inc    = permanence_inc,
            permanence_dec    = permanence_dec,)

        self.reset()

    def reset(self):
        self.active.zero()
        self.learning.zero()

    def compute(self,
        column_sdr=None,
        context_sdr=None,
        context_learning_sdr=None,
        learn=True,):
        """
        Attribute anomaly, mean_anomaly are the fraction of column activations
                  which were predicted.  Range [0, 1]
        """
        ########################################################################
        # PHASE 1:  Make predictions based on the previous timestep.
        ########################################################################
        if context_sdr is None:
            context_sdr = self.active
        excitement, potential    = self.synapses.compute(input_sdr=context_sdr)
        self.excitement          = excitement
        self.predictive_segments = self.excitement >= self.predictive_threshold
        self.predictions         = np.sum(self.predictive_segments, axis=2)
        self.potential_excitement = potential
        self.matching_segments    = potential >= self.learning_threshold

        ########################################################################
        # PHASE 2:  Determine the currently active neurons.
        ########################################################################
        self.columns.assign(column_sdr)
        columns = self.columns.flat_index

        # Activate all neurons which are in a predictive state and in an active
        # column.
        active_dense      = self.predictions[columns]
        col_num, neur_idx = np.nonzero(active_dense)
        # This gets the actual column index, undoes the effect of discarding the
        # inactive columns before the nonzero operation.  
        col_idx           = columns[col_num]
        predicted_active  = (col_idx, neur_idx)
        predicted_values  = np.full(len(predicted_active[0]), self.predicted_boost, dtype=np.uint8)

        # If a column activates but was not predicted by any neuron segment,
        # then it bursts.  The bursting columns are the unpredicted columns.
        bursting_columns = np.setdiff1d(columns, col_idx)
        # All neurons in bursting columns activate.
        burst_col_idx  = np.repeat(bursting_columns, self.cells_per_column)
        burst_neur_idx = np.tile(np.arange(self.cells_per_column), len(bursting_columns))
        burst_active   = (burst_col_idx, burst_neur_idx)
        burst_values   = np.ones(len(burst_active[0]), dtype=np.uint8)

        self.active.index = tuple(np.concatenate([predicted_active, burst_active], axis=1))
        self.active.nz_values = np.concatenate([predicted_values, burst_values])

        # Anomally metric.
        self.anomaly      = len(bursting_columns) / len(columns)
        alpha             = self.anomaly_alpha
        self.mean_anomaly = (1-alpha)*self.mean_anomaly + alpha*self.anomaly

        ########################################################################
        # PHASE 3:  Learn about the previous to current timestep transition.
        ########################################################################
        if not learn:
            return

        # Set the inputs to the previous winner cells.  Grow synapses from these
        # cells and reinforce synapses from them.
        if context_learning_sdr is None:
            prev_ctx_learning = self.learning
        else:
            1/0 # Unimplemented.
        self.synapses.input_sdr.assign(prev_ctx_learning)

        self._learn_predicted(predicted_active)
        bursting_learning = self._learn_bursting(bursting_columns)
        self._learn_mispredicted(columns)

        self.learning.index = tuple(np.concatenate([predicted_active, bursting_learning], axis=1))

    def _learn_predicted(self, predicted_active):
        # Apply Hebbian learning to all active segments in correctly predicted
        # minicolumns.
        cell_num, seg_idx = np.nonzero(self.predictive_segments[predicted_active])
        col_idx  = predicted_active[0][cell_num]
        cell_idx = predicted_active[1][cell_num]
        segments = (col_idx, cell_idx, seg_idx)
        self.synapses.learn(output_sdr = segments)

        # Add synapses to active segments until they have self.add_synapses many
        # active presynapses.
        self.synapses.output_sdr.index     = segments
        self.synapses.output_sdr.nz_values = np.array(np.maximum(0, self.add_synapses - self.potential_excitement[segments]), dtype=np.uint8)
        self.synapses.add_synapses(
            maximum_new_synapses = self.add_synapses,
            maximum_synapses     = self.synapses_per_segment,)

    def _learn_bursting(self, bursting_columns):
        # Select a single segment to learn in every bursting column.

        # First look for subthreshold matching segments.
        matching = self.matching_segments[bursting_columns]

        # Now select a single matching segments for each column.
        columns_need_new_segment = []
        matching_segments        = ([], [], [])
        potential_excitement     = self.potential_excitement
        for col_num, col_idx in enumerate(bursting_columns):
            neur_idx, seg_idx = np.nonzero(matching[col_num])
            num_matches = len(neur_idx)
            if num_matches == 0:
                columns_need_new_segment.append(col_idx)
            else:
                # TODO: Break ties randomly.
                match_idx = np.argmax(potential_excitement[col_idx, neur_idx, seg_idx])
                matching_segments[0].append(col_idx)
                matching_segments[1].append(neur_idx[match_idx])
                matching_segments[2].append(seg_idx[match_idx])

        if columns_need_new_segment:
            # If there are no matching segments, create a segment on the least used
            # neuron (by number of populated segments) and on the least used segment
            # (by potential synapse count).
            # Determine how many potential synapses each segment has.
            sources       = self.synapses.postsynaptic_sources.reshape(self.synapses.output_sdr.dimensions)
            arr_sz        = lambda pp: pp.shape[0]
            source_counts = np.frompyfunc(arr_sz, 1, 1)(sources[columns_need_new_segment])
            # Find the neuron with the fewest populated segments in each column.
            segs_per_neuron   = np.count_nonzero(source_counts, axis=2)
            segs_per_neuron   = segs_per_neuron + np.random.random(size=segs_per_neuron.shape) # Break ties randomly.
            least_used_neuron = np.argmin(segs_per_neuron, axis=1)
            # Find the segments with the fewest synapses.
            source_counts = source_counts[np.arange(source_counts.shape[0]), least_used_neuron]
            source_counts = source_counts + np.random.random(size=source_counts.shape) # Break ties randomly.
            least_used_segment = np.argmin(source_counts, axis=1)
            new_segments = (columns_need_new_segment, least_used_neuron, least_used_segment)
            burst_learning = np.concatenate([matching_segments, new_segments], axis=1)
            burst_learning = tuple(np.array(burst_learning, dtype=np.int, copy=False))
        else:
            burst_learning = tuple(np.array(matching_segments, dtype=np.int))

        # Apply the Hebbian learning rules.
        self.synapses.learn(output_sdr = burst_learning,)

        # Create additional synapses.  Add synapses to the segments until they
        # have self.add_synapses many active presynapses.
        self.synapses.output_sdr.index     = burst_learning
        self.synapses.output_sdr.nz_values = np.array(self.add_synapses - self.potential_excitement[burst_learning], dtype=np.uint8)
        self.synapses.add_synapses(
            maximum_new_synapses = self.add_synapses,
            maximum_synapses     = self.synapses_per_segment,)

        # Return a list of neurons which are learning.  Strip off the segments
        # dimension.
        return burst_learning[0:2]

    def _learn_mispredicted(self, active_columns):
        """
        All matching/learning segments in inactive mincolumns receive a small
        permanence penalty.  This penalizes only the synapses with active
        presynapses.
        """
        # mispredictions = np.array(self.predictive_segments, copy=True)
        mispredictions = np.array(self.matching_segments, copy=True)
        mispredictions[active_columns] = 0
        mispredictions.dtype = np.uint8
        self.synapses.learn(output_sdr     = mispredictions,
                            permanence_inc = -self.mispredict_dec,
                            permanence_dec = 0,)

    def statistics(self):
        stats  = 'Temporal Memory\n'
        stats += self.synapses.statistics()

        # TODO: These should report both the individual neuron and the
        # population wide error rates.  (multiply by pop size?)
        stats += '\tTheoretic False Positive Rate  {:g}\n'.format(
            float(self.synapses.input_sdr.false_positive_rate(
                self.add_synapses,
                self.predictive_threshold,
            ))
        )
        for noise in [5, 10, 20, 50]:
            stats += '\tTheoretic False Negative Rate, {}% noise, {:g}\n'.format(
                noise,
                float(self.synapses.input_sdr.false_negative_rate(
                    noise/100.,
                    self.add_synapses,
                    self.predictive_threshold,
                ))
            )

        stats += "Mean anomaly %d%%\n"%int(round(100 * self.mean_anomaly))
        stats += 'Activation statistics ' + self.active.statistics()
        return stats