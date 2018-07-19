# Written by David McDougall, 2018
# cython: language_level=3

DEF DEBUG = False
debug = DEBUG
print("DEBUG:", 'ON' if DEBUG else 'OFF')

import numpy as np
cimport numpy as np
cimport cython
import array
import math
import scipy.stats
import copy
from sdr import SDR

ctypedef np.float32_t PERMANENCE_t  # For Cython's compile time type system.
PERMANENCE = np.float32             # For Python3's run time type system.
# Note:   "<type?>" is safe case, "<type>" is unsafe cast.

ctypedef np.uint32_t INDEX_t
INDEX = np.uint32

cdef _min(int a, int b):
    if a < b:
        return a
    return b

class SynapseManager:
    """
    This class models a set of synapses between two SDRs.

    Internal to this class, all inputs and outputs are identified by their index
    into their flattened space.  This class indexes its synapses by both the
    presynaptic input index and the postsynaptic output index.  Each output
    keeps a list of potential inputs, each input keeps a list of potential
    outputs, and both sides contain the complete location of their other
    endpoint.  In this way every input and output can quickly access all of its
    data.

    Attributes postsynaptic_sources, postsynaptic_permanences
        postsynaptic_sources[output-index] = 1D array of potential input indices.
        postsynaptic_permanences[output-index] = 1D array of potential input permanences.
        These tables run parallel to each other.  These tables are the original
        data tables; the other tables are calculated from these two tables by
        the method rebuild_indexes().  These two tables specify the pool of
        potential and connected input connections to each output location, and
        are refered to as the "potential_pool".

    Attibute postsynaptic_sources_sizes
        postsynaptic_sources_sizes[output-index] = The logical size of the 
        postsnyaptic_sources entry.  The postsynaptic tables are allocated with
        extra room at the end for fast adding and removing synapses.

    Attribute postsynaptic_connected_count
        postsynaptic_connected_count[output-index] = The number of synapses
        which connect to this output site.  Values in this table can be computed
        with the following code snippet:
            output-index = 1234
            output-size  = sm.postsynaptic_sources_sizes[output-index]
            permanences  = sm.postsynaptic_permanences[ : output-size]
            connected    = permanences >= sm.permanence_thresh
            postsynaptic_connected_count[output-index] = np.count_nonzero(connected)

    Attributes presynaptic_sinks, presynaptic_sinks_sizes
        presynaptic_sinks[input-index] = 1D array of potential output indices.
        The sinks table is associates each input (presynapse) with its outputs
        (postsynapses), allowing for fast feed forward calculations.  
        presynaptic_sinks_sizes[input-index] = The logical size of the
        presynaptic_sinks entry.  The presynaptic_sinks are  allocated with
        extra space to facilitate adding and removing synapses.

    Attribute presynaptic_partitions
        presynapic_partitions[input-index] = Number of connected synapses this
        input makes. The presynaptic_sinks table is partitioned by permanence
        value such that the first synapses in the table are connected and the
        remaining are disconnected.

    Since postsynaptic_sources and presynaptic_sinks are arrays of arrays, two
    indices are needed to go back and forth between them.  Attributes
    postsynaptic_source_side_index and presynaptic_sink_side_index are tables
    containing these second indexes.

    Attribute postsynaptic_source_side_index
        This table runs parallel to the postsynaptic_sources table and contains
        the second index into the presynaptic_sinks table.  Values in this table
        can be derived from the following pseudocode:
            output_index         = 1234
            potential_pool_index = 66
            input_index          = postsynaptic_sources[output_index][potential_pool_index]
            source_side_index    = presynaptic_sinks[input_index].index(output_index)  # See help(list.index)
            postsynaptic_source_side_index[output_index][potential_pool_index] = source_side_index

    Attribute presynaptic_sink_side_index
        This table runs parallel to presynaptic_sinks.  It contains the second
        index into the postsynaptic_sources table.  Values in this table can be
        derived from the following pseudocode:
            input_index     = 12345
            sink_number     = 99
            output_index    = presynaptic_sinks[input_index][sink_number]
            potential_index = postsynaptic_sources[output_index].index(input_index)
            presynaptic_sink_side_index[input_index][sink_number] = potential_index

    Attribute presynaptic_permanences contain the synapse permanences, indexed
        from the input side.  This attibute is only available if the
        SynapseManager was created with weighted = True.

    Data types:
        Global variable "INDEX" is np.uint32.
        Global variable "PERMANENCE" is np.float32.

        Network Size Assumptions:
        sm.input_sdr.size  <= 2^32, input space is addressable by type INDEX.
        sm.output_sdr.size <= 2^32, output space is addressable by type INDEX.

        sm.postsynaptic_sources.dtype == INDEX, index into input space.

        sm.postsynaptic_sources_sizes.dtype == INDEX

        sm.postsynaptic_connected_count.dtype == INDEX

        sm.postsynaptic_source_side_index.dtype == INDEX, maximum number of 
            outputs which an input could connect to, size of output space.

        sm.presynaptic_sinks.dtype == INDEX, index into output space.

        sm.presynaptic_sinks_sizes.dtype == INDEX, maximum number of outputs
            which an input could connect to at one time, size of output space.

        sm.presynaptic_sink_side_index.dtype == INDEX, maximum number of inputs
            which an output could have in its potential pool, size of input
            space.

        sm.postsynaptic_permanences.dtype == PERMANENCE.
        sm.presynaptic_permanences.dtype  == PERMANENCE.
        sm.self.permanence_inc     is type PERMANENCE.
        sm.self.permanence_dec     is type PERMANENCE.
        sm.self.permanence_thresh  is type PERMANENCE.
        NOTE: Synapses are connected if and only if: permanence >= threshold.
        And both variables MUST use the correct types!
    """
    def __init__(self, input_sdr, output_sdr, radii, init_dist,
        permanence_thresh      = 0,
        permanence_inc         = 0,
        permanence_dec         = 0,
        initial_potential_pool = 0,
        weighted               = False,):
        """
        Argument input_sdr is the presynaptic input activations.
        Argument output_sdr is the postsynaptic segment sites.

        Argument init_dist is (mean, std) of permanence value random initial distribution.
        Optional Argument permanence_thresh ...
        Optional Argument permanence_inc ...
        Optional Argument permanence_dec ...


        Topology:  Makes synapses from inputs to outputs within their local
        neighborhood. The outputs exist on a uniform grid which is stretched
        over the input space.  An outputs local neighborhood is defined by a
        gaussian window centered over the output with standard deviations given
        by argument radii.

        Argument initial_potential_pool is the number of potential inputs to
            connect each output to.  Optional: by default this is 0.  More
            synapses can be added after initialization with the SM.add_synapses
            method.

        Argument radii is tuple of standard deivations. Radii units are the
                 input space units.  Radii defines the topology of the
                 connections.  If radii are shorter than the number of input or
                 output space dimensions then the trailing input dimensions are
                 not considered topological dimensions.  These 'extra'
                 dimensions are treated with uniform probability; only distances
                 in the topological dimensions effect the probability of forming
                 a potential synapse.


        Optional Argument weighted is a bool.
            The compute_weighted method uses the synapse permancence as a
            connection strength, instead of as a boolean connection.
            ...  synapse strength to the excitement accumulator instead of incrementing it.
            The  learn method still modifies the permanence using hebbian
            learning.

        """
        assert(isinstance(input_sdr, SDR))
        assert(isinstance(output_sdr, SDR))
        self.input_sdr         = input_sdr
        self.output_sdr        = output_sdr
        self.radii             = tuple(radii) if radii is not None else None
        self.init_dist         = tuple(PERMANENCE(z) for z in init_dist)
        self.permanence_inc    = PERMANENCE(permanence_inc)
        self.permanence_dec    = PERMANENCE(permanence_dec)
        self.permanence_thresh = PERMANENCE(permanence_thresh)
        self.weighted          = bool(weighted)

        assert(self.input_sdr.size  <= 2**32)   # self.postsynaptic_sources is a uint32 index into input space
        assert(self.output_sdr.size <= 2**32)   # self.presynaptic_sinks is a uint32 index into output space

        if self.radii is not None and len(self.radii):
            (self.postsynaptic_locations,
            self.postsynaptic_source_distributions) = _setup_topology(
                self.input_sdr.dimensions,
                self.output_sdr.dimensions,
                self.radii,)

        self.postsynaptic_sources       = np.empty(self.output_sdr.size, dtype=object)
        self.postsynaptic_permanences   = np.empty(self.output_sdr.size, dtype=object)
        for idx in range(self.output_sdr.size):
            self.postsynaptic_sources[idx]     = np.empty((0,), dtype=INDEX)
            self.postsynaptic_permanences[idx] = np.empty((0,), dtype=PERMANENCE)
        self.rebuild_indexes()  # Initializes sinks index and the rest.

        if initial_potential_pool > 0:
            self.add_synapses(
                input_sdr            = np.ones(self.input_sdr.dimensions, dtype=np.uint8),
                output_sdr           = np.ones(self.output_sdr.dimensions, dtype=np.uint8),
                num_synapses         = np.full(self.output_sdr.size, initial_potential_pool, dtype=np.int),
                maximum_new_synapses = initial_potential_pool,
                maximum_synapses     = initial_potential_pool,
                init_dist            = init_dist,)

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(False)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def compute(self, input_sdr=None):
        """
        Applies presynaptic activity to synapses, returns the postsynaptic
        excitment.

        Argument input_sdr ... is assigned to this classes internal inputs SDR.
                 If not given this uses the current value of its inputs SDR,
                 which this synapse manager was initialized with.

        Returns the pair (excitement, potential_excitement)
            Both are np.ndarray, shape == output_sdr.dimensions, dtype == INDEX
            excitement contains ...
            potential_excitement contains the excitement from both connected and
            disconnected synapses.
        """
        self.input_sdr.assign(input_sdr)
        cdef:
            # Data Tables.
            np.ndarray[object]       sinks_table = self.presynaptic_sinks
            np.ndarray[INDEX_t]      sink_sizes  = self.presynaptic_sinks_sizes
            np.ndarray[INDEX_t]      sink_parts  = self.presynaptic_partitions
            # Arguments and Return Values.
            np.ndarray[np.uint8_t]   inps        = self.input_sdr.dense.reshape(-1)
            np.ndarray[INDEX_t]      excitement  = np.zeros(self.output_sdr.size, dtype=INDEX)
            np.ndarray[INDEX_t]      potential_x = np.zeros(self.output_sdr.size, dtype=INDEX)
            # Locals and Inner Array Pointers.
            np.ndarray[INDEX_t]      sinks_inner
            INDEX_t inp_idx, inp_idx2, out_idx, partition
            np.uint8_t inp_value

        # For each active input.
        for inp_idx in range(inps.shape[0]):
            inp_value = inps[inp_idx]
            if inp_value == 0:
                continue

            sinks_inner = <np.ndarray[INDEX_t]> sinks_table[inp_idx]
            # Tally the connected synapses.
            partition = sink_parts[inp_idx]
            for inp_idx2 in range(partition):
                out_idx = sinks_inner[inp_idx2]
                excitement[out_idx]  += inp_value
                potential_x[out_idx] += 1
            # Tally the potential synapses.
            for inp_idx2 in range(partition, sink_sizes[inp_idx]):
                out_idx = sinks_inner[inp_idx2]
                potential_x[out_idx] += 1

        return (excitement.reshape(self.output_sdr.dimensions),
                potential_x.reshape(self.output_sdr.dimensions))

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(False)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def compute_weighted(self, input_sdr=None):
        """
        Applies presynaptic activity to synapses, returns the postsynaptic
        excitment.

        Argument input_sdr ... is assigned to this classes internal inputs SDR.
                 If not given this uses the current value of its inputs SDR,
                 which this synapse manager was initialized with.

        Returns the excitement ... shape is output_sdr.dimensions
        """
        1/0 # i broke it...
        assert(self.weighted)
        self.input_sdr.assign(input_sdr)
        cdef:
            np.ndarray[np.uint8_t]    inputs        = self.input_sdr.dense.reshape(-1)
            np.ndarray[object]        sinks_table   = self.presynaptic_sinks
            np.ndarray[INDEX_t]       sinks_entry
            np.ndarray[object]        perms_table   = self.presynaptic_permanences
            np.ndarray[INDEX_t]       sink_sizes    = self.presynaptic_sinks_sizes
            np.ndarray[PERMANENCE_t]  perms_entry
            np.ndarray[np.float32_t]  excitement    = np.zeros(self.output_sdr.size, dtype=PERMANENCE)
            INDEX_t inp_idx1, inp_idx2, out_idx
            np.uint8_t inp_value

        for inp_idx1 in range(inputs.shape[0]):
            inp_value = inputs[inp_idx1]
            if inp_value == 0:
                continue

            sinks_entry = <np.ndarray[INDEX_t]>      sinks_table[inp_idx1]
            perms_entry = <np.ndarray[PERMANENCE_t]> perms_table[inp_idx1]

            for inp_idx2 in range(sink_sizes[inp_idx1]):
                out_idx = sinks_entry[inp_idx2]
                excitement[out_idx] += perms_entry[inp_idx2] * inp_value

        return excitement.reshape(self.output_sdr.dimensions)

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(False)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def learn(self, input_sdr=None, output_sdr=None,
        permanence_inc = None,
        permanence_dec = None,
        prev_updates   = None,):
        """
        Update the permanences of active outputs using the most recently given
        inputs.

        Argument output_sdr contains the indices of the outputs to apply hebbian
            learning to.  Output_sdr.nz_values is intentionally not used as
            whether an output learns is a binary choice, no multipliers for
            learning rates are applied.

        Argument input_sdr ... contains active presynaptic inputs ...
            Input_sdr.nz_values is intentionally not used as whether an input
            learns is a binary choice, no multipliers for learning rates are
            applied.

        Optional Arguments permanence_inc and permanence_dec take precedence
           over any value passed to this classes initialize method.

       Optional Argument prev_updates ...
        """
        self.input_sdr.assign(input_sdr)
        self.output_sdr.assign(output_sdr)
        cdef:
            # Data tables
            np.ndarray[object]      sources          = self.postsynaptic_sources
            np.ndarray[object]      sources2         = self.postsynaptic_source_side_index
            np.ndarray[object]      permanences      = self.postsynaptic_permanences
            np.ndarray[INDEX_t]     sources_sizes    = self.postsynaptic_sources_sizes
            np.ndarray[INDEX_t]     connected_counts = self.postsynaptic_connected_count
            np.ndarray[object]      sinks            = self.presynaptic_sinks
            np.ndarray[object]      sinks2           = self.presynaptic_sink_side_index
            np.ndarray[object]      presyn_perms     = getattr(self, 'presynaptic_permanences', None)
            np.ndarray[INDEX_t]     sinks_parts      = self.presynaptic_partitions
            np.ndarray[np.int_t]    output_activity  = self.output_sdr.flat_index
            np.ndarray[np.uint8_t]  input_activity   = self.input_sdr.dense.reshape(-1)

            # Inner array pointers
            np.ndarray[INDEX_t]      sources_inner
            np.ndarray[INDEX_t]      sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[PERMANENCE_t] presyn_perms_inner = None

            # Indexes and locals
            INDEX_t out_size
            INDEX_t out_iter, out_idx1, out_idx2
            INDEX_t inp_idx1, inp_idx2, inp_idx2_swap
            INDEX_t partition
            np.uint8_t input_value
            PERMANENCE_t perm_value
            PERMANENCE_t perm_delta
            PERMANENCE_t inc, dec, thresh = self.permanence_thresh
            bint syn_prior, syn_post

        # Arguments override initialized or default values.
        inc = permanence_inc if permanence_inc is not None else self.permanence_inc
        dec = permanence_dec if permanence_dec is not None else self.permanence_dec
        if inc == 0. and dec == 0.:
            return

        # Correlate input learning stuff.
        cdef:
            np.ndarray[object] _prev_updates
            np.ndarray[object] current_updates = np.full(self.output_sdr.size, None)
            np.ndarray[PERMANENCE_t] prev_update_row
            np.ndarray[PERMANENCE_t] current_update_row
            np.ndarray[object]       prev_updates_merged, update_table
            np.ndarray[PERMANENCE_t] update_row
        if isinstance(prev_updates, tuple):
            # If given a tuple then each element is a complete updates table.
            # Add the update tables, None's are implicit rows of zeros.
            prev_updates_merged = np.full(self.output_sdr.size, None)
            for update_table in prev_updates:
                for out_idx1 in range(self.output_sdr.size):
                    update_row = update_table[out_idx1]
                    if update_row is not None:
                        prev_update_row = prev_updates_merged[out_idx1]
                        if prev_update_row is not None:
                            prev_updates_merged[out_idx1] = (update_row + prev_update_row)
                        else:
                            prev_updates_merged[out_idx1] = update_row
            _prev_updates = prev_updates_merged
        elif prev_updates is not None:
            _prev_updates = prev_updates
        else:
            _prev_updates = current_updates # This is a hack...

        for out_iter in range(output_activity.shape[0]):
            out_idx1 = output_activity[out_iter]

            sources_inner    = <np.ndarray[INDEX_t]>      sources[out_idx1]
            sources2_inner   = <np.ndarray[INDEX_t]>      sources2[out_idx1]
            perms_inner      = <np.ndarray[PERMANENCE_t]> permanences[out_idx1]
            out_size         = sources_sizes[out_idx1]

            current_update_row = <np.ndarray[PERMANENCE_t]> np.zeros(out_size, dtype=PERMANENCE)
            current_updates[out_idx1] = current_update_row  # Possible reference to _prev_updates
            prev_update_row    = <np.ndarray[PERMANENCE_t]> _prev_updates[out_idx1]
            if prev_update_row is None:
                prev_update_row = <np.ndarray[PERMANENCE_t]> np.zeros(out_size, dtype=PERMANENCE)
            elif prev_update_row.shape[0] < out_size:
                # Synapses were added since updates were computed, assign zeros
                # to the new synapses previous updates.
                update_row = <np.ndarray[PERMANENCE_t]> np.zeros(out_size, dtype=PERMANENCE)
                update_row[ : prev_update_row.shape[0]] = prev_update_row
                prev_update_row = update_row

            for out_idx2 in range(out_size):
                inp_idx1     = sources_inner[out_idx2]
                perm_value   = perms_inner[out_idx2]
                syn_prior    = perm_value >= thresh
                # Hebbian Learning.
                input_value  = input_activity[inp_idx1]
                if input_value != 0:
                    perm_delta = inc
                else:
                    perm_delta = -dec
                # Update with hebbian learning and subtract away the previous learning update.
                # perm_value += perm_delta - prev_update_row[out_idx2]
                if perm_delta != prev_update_row[out_idx2]:
                    perm_value += perm_delta

                # Clip permanence to [0, 1]
                if perm_value > 1.:
                    perm_value = 1.
                elif perm_value < 0.:
                    perm_value = 0.
                syn_post = perm_value >= thresh
                perms_inner[out_idx2] = perm_value

                current_update_row[out_idx2] = perm_delta

                if presyn_perms is not None:
                    inp_idx2     = sources2_inner[out_idx2]
                    presyn_perms_inner = <np.ndarray[PERMANENCE_t]> presyn_perms[inp_idx1]
                    presyn_perms_inner[inp_idx2] = perm_value

                if syn_prior != syn_post:
                    inp_idx2     = sources2_inner[out_idx2]
                    partition    = sinks_parts[inp_idx1]

                    if syn_post:
                        # Swap this synapse to the start of the disconnected
                        # segment and increment the partition over it.
                        inp_idx2_swap = partition
                        sinks_parts[inp_idx1] += 1
                        # Book keeping, one more connected synapse to this output site.
                        connected_counts[out_idx1] += 1
                    else:
                        # Swap this synapse to the end of the connected segment
                        # and decrement the partition over of.
                        inp_idx2_swap = partition - 1
                        sinks_parts[inp_idx1] -= 1
                        # Book keeping, one fewer connected synapse to this output site.
                        connected_counts[out_idx1] -= 1

                    _swap_inputs(
                        <np.ndarray[INDEX_t]> sinks[inp_idx1],
                        <np.ndarray[INDEX_t]> sinks2[inp_idx1],
                        presyn_perms_inner,
                        sources2,
                        inp_idx2,
                        inp_idx2_swap)
        return current_updates

    # TODO: Compute maximum_new_synapses locally.
    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(False)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def add_synapses(self, maximum_synapses, maximum_new_synapses, num_synapses,
        init_dist  = None,
        input_sdr  = None,
        output_sdr = None,
        evict      = False,):
        """

        Argument input_sdr contains the sources to grow new synapses from.

        Argument output_sdr contains the outputs to add synapses to.  

        Argument num_synapses contains the number of synapses to add to each 
            output specified in output_sdr.  It runs parallel to
            output_sdr.index and output_sdr.flat_index.

        Argument maximum_synapses which a segment can have.

        Argument maximum_new_synapses ...

        Argument init_dist is (mean, std) of synapse initial permanence
            distribution.   Optional: If not given this will use the values the
            synapse manager was initialized with.

        Optional Argument evict is boolean, default False.  If true this will
            make room for new syanpases on full segments by removing some of the
            synapses with the minimum permanences.
        """
        # Arguments.
        self.input_sdr.assign(input_sdr)
        self.output_sdr.assign(output_sdr)
        if len(self.input_sdr) == 0 or len(self.output_sdr) == 0:
            return
        cdef:
            np.ndarray[INDEX_t]   inputs    = np.array(self.input_sdr.flat_index, dtype=INDEX)
            np.ndarray[np.long_t] outputs   = self.output_sdr.flat_index
            np.ndarray[np.int_t]  new_syns  = np.array(num_synapses, dtype=np.int)
        assert(len(new_syns) == len(outputs))
        if init_dist is not None:
            init_mean, init_std = init_dist
        else:
            init_mean, init_std = self.init_dist

        cdef:            # Data Tables.
            np.ndarray[object] sources       = self.postsynaptic_sources
            np.ndarray[object] sources2      = self.postsynaptic_source_side_index
            np.ndarray[object] permanences   = self.postsynaptic_permanences
            np.ndarray[INDEX_t] sources_sizes = self.postsynaptic_sources_sizes
            np.ndarray[object] presyn_perms  = getattr(self, 'presynaptic_permanences', None)
            np.ndarray[object] sinks         = self.presynaptic_sinks
            np.ndarray[object] sinks2        = self.presynaptic_sink_side_index
            np.ndarray[INDEX_t] sinks_sizes  = self.presynaptic_sinks_sizes
            np.ndarray[INDEX_t] sinks_parts  = self.presynaptic_partitions
            np.ndarray[object] source_dist   = getattr(self, 'postsynaptic_source_distributions', None)
        cdef:            # Inner array pointers.
            np.ndarray[INDEX_t]      sources_inner
            np.ndarray[INDEX_t]      sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[INDEX_t]      sinks_inner
            np.ndarray[INDEX_t]      sinks2_inner
            np.ndarray[PERMANENCE_t] presyn_perms_inner = None
        cdef:            # Locals indexes and counters.
            int out_iter, out_idx, out_idx2
            INDEX_t inp_idx1, inp_idx2
            np.ndarray[INDEX_t]      candidate_sources
            np.ndarray[INDEX_t]      new_sources
            np.ndarray[PERMANENCE_t] new_permanances
            PERMANENCE_t perm, thresh = self.permanence_thresh
            np.int32_t destination_id
            np.ndarray[np.int32_t] candidate_ids
            int max_syn     = maximum_synapses
            int max_new_syn = min(maximum_new_synapses, inputs.shape[0])
            int nsyn_to_add
            INDEX_t orig_nsyn, num_new_syns
            # Evict min perm synapses variables.
            bint evict_ = bool(evict)
            int overrun, evict_iter, out_idx2_remove
            np.ndarray[INDEX_t] evict_synapses
            # Topology variables.
            int topo_area_size
            int topo_out_idx, next_topo_area
            np.ndarray[np.double_t] input_p = None

            int buff_size
            np.ndarray[INDEX_t]      index_buffer = np.full(max_new_syn, -1,           dtype=INDEX)
            np.ndarray[PERMANENCE_t] perms_buffer = np.full(max_new_syn, float('nan'), dtype=PERMANENCE)

        # Setup topology.
        if source_dist is not None:
            nd_topological = len(self.radii)
            topo_area_size = np.product(self.output_sdr.dimensions[nd_topological : ])
            topo_inp_index = self.input_sdr.index[ : nd_topological]
            next_topo_area = 0
            IF DEBUG:
                output_topo_areas = outputs // topo_area_size
                assert(np.all(output_topo_areas == np.sort(output_topo_areas)))
                assert(np.all(new_syns >= 0))         # Surely an error, check for unsigned less than zero mistakes.
                assert(np.all(new_syns <= max_syn))   # Surely an error, check for unsigned less than zero mistakes.
        else:
            next_topo_area = self.output_sdr.size

        for out_iter in range(outputs.shape[0]):
            out_idx     = outputs[out_iter]
            nsyn_to_add = new_syns[out_iter]
            orig_nsyn   = sources_sizes[out_idx]

            # Can not add more synapses than there are available inputs for.
            nsyn_to_add = _min(nsyn_to_add, max_new_syn)
            # Make room on the segment for more synapses by removing minimum
            # permanence synapses, if desired.
            overrun = orig_nsyn + nsyn_to_add - max_syn
            if evict_ and overrun > 0:
                sources_inner  = <np.ndarray[INDEX_t]>  sources[out_idx]
                perms_inner    = <np.ndarray[PERMANENCE_t]> permanences[out_idx]
                evict_synapses = _find_min_perm_synapses(
                    overrun,
                    sources_inner,
                    perms_inner,
                    orig_nsyn,
                    inputs,)
                # Removing synapses changes where all of the synapses are, so
                # sort and remove from the ends of the postsynaptic tables so
                # that the evict_synapses stay in the same place until they're
                # removed.  
                evict_synapses.sort()
                for evict_iter in range(evict_synapses.shape[0] - 1, -1, -1):
                    out_idx2_remove = evict_synapses[evict_iter]
                    _remove_synapse(
                        sources,
                        sources2,
                        permanences,
                        sources_sizes,
                        sinks,
                        sinks2,
                        presyn_perms,
                        sinks_parts,
                        sinks_sizes,
                        thresh,
                        out_idx,
                        out_idx2_remove,)
                orig_nsyn = sources_sizes[out_idx]
            # Enforce the maximum number of synapses per output.
            nsyn_to_add = _min(nsyn_to_add, max_syn - orig_nsyn)
            if nsyn_to_add == 0:
                continue

            # Reload these inner data tables after evicting synapses B/C they've
            # changed.
            sources_inner     = <np.ndarray[INDEX_t]>  sources[out_idx]
            sources2_inner    = <np.ndarray[INDEX_t]>  sources2[out_idx]
            perms_inner       = <np.ndarray[PERMANENCE_t]> permanences[out_idx]

            # Check if out_iter has moved to a new topological area.
            if out_idx >= next_topo_area:
                topo_out_idx   = out_idx // topo_area_size
                next_topo_area = (topo_out_idx + 1) * topo_area_size
                input_p = source_dist[topo_out_idx][topo_inp_index]
                input_p = input_p / np.sum(input_p)

            # Randomly select inputs.
            candidate_sources = np.random.choice(inputs,
                                    size    = nsyn_to_add,
                                    replace = False,
                                    p       = input_p,)
            # Filter out inputs which have already been connected to.
            unique_sources    = np.isin(candidate_sources, sources_inner, invert=True,
                                        assume_unique = True,)
            new_sources = candidate_sources[unique_sources]
            num_new_syns = new_sources.shape[0]

            # Random initial distribution of permanence values.
            new_permanances_f64 = np.random.normal(init_mean, init_std, size=num_new_syns)
            new_permanances     = np.array(new_permanances_f64, dtype=PERMANENCE)
            np.clip(new_permanances, 0., 1., out=new_permanances)

            # Resize the sources tables to fit the new synapses.
            if orig_nsyn + num_new_syns > sources_inner.shape[0]:
                # Don't allocate room for more synapses than could possibly fit
                # on the segment.  buff_size slices off useful portion of buffer.
                buff_size = _min(max_syn - sources_inner.shape[0], max_new_syn)
                sources[out_idx]     = sources_inner  = np.append(sources_inner,  index_buffer[:buff_size])
                sources2[out_idx]    = sources2_inner = np.append(sources2_inner, index_buffer[:buff_size])
                permanences[out_idx] = perms_inner    = np.append(perms_inner,    perms_buffer[:buff_size])
            # Note: num_new_syns changes meanings: from number of synapses to
            # be added, to total number of synapses after adding them.
            num_new_syns += orig_nsyn
            sources_sizes[out_idx] = num_new_syns
            # Append to the postsynaptic source tables.
            sources_inner[orig_nsyn : num_new_syns] = new_sources
            perms_inner[  orig_nsyn : num_new_syns] = new_permanances

            # Process each new synapse into the data tables.
            for out_idx2 in range(orig_nsyn, num_new_syns):
                inp_idx1  = sources_inner[out_idx2]
                perm      = perms_inner[out_idx2]

                # Gather data tables.
                sinks_inner  = <np.ndarray[INDEX_t]> sinks[inp_idx1]
                sinks2_inner = <np.ndarray[INDEX_t]> sinks2[inp_idx1]
                if presyn_perms is not None:
                    presyn_perms_inner = <np.ndarray[PERMANENCE_t]> presyn_perms[inp_idx1]
                # Append to the presynaptic sinks tables.
                inp_idx2 = sinks_sizes[inp_idx1]
                sinks_sizes[inp_idx1] += 1
                # Resize presynaptic tables if necessary.
                if inp_idx2 >= sinks_inner.shape[0]:
                    sinks_inner      = np.append(sinks_inner,  index_buffer)
                    sinks[inp_idx1]  = sinks_inner
                    sinks2_inner     = np.append(sinks2_inner, index_buffer)
                    sinks2[inp_idx1] = sinks2_inner
                    if presyn_perms is not None:
                        presyn_perms_inner     = np.append(presyn_perms_inner, perms_buffer)
                        presyn_perms[inp_idx1] = presyn_perms_inner
                # Insert the synapse.
                sinks_inner[inp_idx2]  = out_idx
                sinks2_inner[inp_idx2] = out_idx2
                sources2_inner[out_idx2] = inp_idx2 # Link sources back to sinks.
                if presyn_perms is not None:
                    presyn_perms_inner[inp_idx2] = perm

                if perm >= thresh:
                    # Synapse is initially connected.  Swap it in the input
                    # table to the first synapse in the disconnected section and
                    # then increment the partition over it.
                    _swap_inputs(
                        <np.ndarray[INDEX_t]> sinks_inner,
                        <np.ndarray[INDEX_t]> sinks2_inner,
                        presyn_perms_inner,
                        sources2,
                        inp_idx2,
                        sinks_parts[inp_idx1])
                    sinks_parts[inp_idx1] += 1

    @cython.profile(DEBUG)
    def rebuild_indexes(self):
        """
        This method uses the postsynaptic_sources and postsynaptic_permanences
        tables to rebuild all of the other needed tables.

        Notice: this does NOT use attribute postsynaptic_sources_sizes, this
        assumes that every synapse in the potential pool is valid.
        """
        cdef:
            # Source Data Tables
            np.ndarray[object]  sources       = self.postsynaptic_sources
            np.ndarray[object]  permanences   = self.postsynaptic_permanences
            # New Data Tables.
            np.ndarray[object]  sources2      = np.empty(self.output_sdr.size, dtype=object)
            np.ndarray[INDEX_t] sources_sizes = np.zeros(self.output_sdr.size, dtype=INDEX)
            np.ndarray[INDEX_t] con_counts    = np.zeros(self.output_sdr.size, dtype=INDEX)
            np.ndarray[object]  sinks         = np.empty(self.input_sdr.size,  dtype=object)
            np.ndarray[object]  sinks2        = np.empty(self.input_sdr.size,  dtype=object)
            np.ndarray[INDEX_t] sinks_sizes   = np.zeros(self.input_sdr.size,  dtype=INDEX)
            np.ndarray[INDEX_t] sinks_parts   = np.zeros(self.input_sdr.size,  dtype=INDEX)
            np.ndarray[object]  sink_perms    = np.empty(self.input_sdr.size,  dtype=object)
            # Inner array pointers
            np.ndarray[INDEX_t]  sources_inner
            np.ndarray[INDEX_t]  sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            # Locals
            int inp_idx, inp_idx2, out_idx, out_idx2, synapse_num
            int num_sources
            PERMANENCE_t perm_val, thresh = self.permanence_thresh
        index_buffer = np.full(20, -1,           dtype=INDEX)
        perms_buffer = np.full(20, float('nan'), dtype=PERMANENCE)
        # Initialize the index tables with python lists for fast append.
        for inp_idx in range(self.input_sdr.size):
            sinks[inp_idx]      = array.array('L')
            sinks2[inp_idx]     = array.array('L')
            sink_perms[inp_idx] = array.array('f')
        # Iterate through every synapse from the output side, build the input
        # side tables.
        for out_idx in range(self.output_sdr.size):
            sources_inner          = sources[out_idx]
            perms_inner            = permanences[out_idx]
            num_sources            = len(sources_inner)
            sources_sizes[out_idx] = num_sources
            sources2[out_idx]      = np.full(num_sources, -1, dtype=INDEX)
            for synapse_num in range(num_sources):
                inp_idx  = sources_inner[synapse_num]
                perm_val = perms_inner[synapse_num]
                assert(perm_val >= 0. and perm_val <= 1.)
                if perm_val >= thresh:
                    con_counts[out_idx] += 1
                sinks[inp_idx].append(out_idx)
                sinks2[inp_idx].append(synapse_num)
                sink_perms[inp_idx].append(perm_val)
        # Iterate through every synapse from the input side.  
        for inp_idx in range(self.input_sdr.size):
            # Cast input side tables to numpy arrays.
            sinks_inner  = np.array(sinks[inp_idx],      dtype=INDEX)
            sinks2_inner = np.array(sinks2[inp_idx],     dtype=INDEX)
            perms_inner  = np.array(sink_perms[inp_idx], dtype=PERMANENCE)
            # Partition presynaptic tables into connected/potential segments.
            part_order   = np.argsort(-perms_inner)
            sinks_inner  = sinks_inner[part_order]
            sinks2_inner = sinks2_inner[part_order]
            perms_inner  = perms_inner[part_order]
            sinks_parts[inp_idx] = np.count_nonzero(perms_inner >= thresh)
            # Notify the postsynaptic side of the final location of this synapse
            # in the presynaptic tables.
            for inp_idx2 in range(sinks_inner.shape[0]):
                out_idx  = sinks_inner[inp_idx2]
                out_idx2 = sinks2_inner[inp_idx2]
                sources2_inner = sources2[out_idx]
                sources2_inner[out_idx2] = inp_idx2
            # Pad the sink side tables with some room to grow.
            sinks_sizes[inp_idx] = len(sinks[inp_idx])
            sinks[inp_idx]       = np.append(sinks_inner,  index_buffer)
            sinks2[inp_idx]      = np.append(sinks2_inner, index_buffer)
            sink_perms[inp_idx]  = np.append(perms_inner,  perms_buffer)
        # Save the data tables.
        self.presynaptic_sinks              = sinks
        self.presynaptic_sink_side_index    = sinks2
        self.presynaptic_sinks_sizes        = sinks_sizes
        self.presynaptic_partitions         = sinks_parts
        self.postsynaptic_source_side_index = sources2
        self.postsynaptic_sources_sizes     = sources_sizes
        self.postsynaptic_connected_count   = con_counts
        if self.weighted:
            self.presynaptic_permanences    = sink_perms


    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(False)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def remove_zero_permanence_synapses(self):
        cdef:
            np.ndarray[object]      sources          = self.postsynaptic_sources
            np.ndarray[object]      sources2         = self.postsynaptic_source_side_index
            np.ndarray[object]      permanences      = self.postsynaptic_permanences
            np.ndarray[INDEX_t]     sources_sizes    = self.postsynaptic_sources_sizes
            np.ndarray[object]      sinks            = self.presynaptic_sinks
            np.ndarray[object]      sinks2           = self.presynaptic_sink_side_index
            np.ndarray[object]      presyn_perms     = getattr(self, 'presynaptic_permanences', None)
            np.ndarray[INDEX_t]     sinks_sizes      = self.presynaptic_sinks_sizes

            np.ndarray[INDEX_t]      sources_inner
            np.ndarray[INDEX_t]      sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[PERMANENCE_t] presyn_perms_inner = None

            int out_idx1, out_idx2
            INDEX_t n_syn
            INDEX_t inp_idx1, inp_idx2
            INDEX_t new_sink_sz
            PERMANENCE_t perm_value

        # Search for synapses to remove.
        for out_idx1 in range(sources.shape[0]):
            sources_inner  = <np.ndarray[INDEX_t]>      sources[out_idx1]
            sources2_inner = <np.ndarray[INDEX_t]>      sources2[out_idx1]
            perms_inner    = <np.ndarray[PERMANENCE_t]> permanences[out_idx1]

            n_syn = sources_sizes[out_idx1]

            out_idx2 = 0
            while out_idx2 < n_syn:
                perm_value = perms_inner[out_idx2]

                if perm_value == 0.:
                    # Delete this synapse.
                    inp_idx1 = sources_inner[out_idx2]
                    inp_idx2 = sources2_inner[out_idx2]

                    # Swap the synapse to the end of the presynaptic/input side
                    # tables and decrement the logical size of the table.
                    new_sink_sz = sinks_sizes[inp_idx1] - 1
                    if presyn_perms is not None:
                        presyn_perms_inner = <np.ndarray[PERMANENCE_t]> presyn_perms[inp_idx1]
                    _swap_inputs(
                        <np.ndarray[INDEX_t]> sinks[inp_idx1],
                        <np.ndarray[INDEX_t]> sinks2[inp_idx1],
                        presyn_perms_inner,
                        sources2,
                        inp_idx2,
                        new_sink_sz)
                    sinks_sizes[inp_idx1] = new_sink_sz

                    # Swap this synapse to the end of the postsynaptic/output
                    # side tables.
                    n_syn -= 1
                    _swap_outputs(
                        sources_inner,
                        sources2_inner,
                        perms_inner,
                        sinks2,
                        out_idx2,
                        n_syn,)
                else:
                    # Only move on to the next synapse if this did NOT just swap
                    # a synapse into the current location.
                    out_idx2 += 1

            # Update the logical size of the output tables so that the
            # removed synapses are effectively gone.
            sources_sizes[out_idx1] = n_syn

    def copy(self):
        """
        Makes a shallow copy of the synapse manager and its input/output SDRs,
        which effectively freezes and disconnects this from the rest of the
        system while sharing the same underlying data tables.  When the copy
        learns, the original also learns.
        """
        cpy = copy.copy(self)
        cpy.inputs  = SDR(self.input_sdr)
        cpy.outputs = SDR(self.output_sdr)
        return cpy

    def input_location(self, input_index):
        """
        Argument input_index is flat_index of presynaptic input site.

        Returns the location of the given input in the input space.
        """
        nd_topological  = len(self.radii)
        topo_inp_dims   = self.input_sdr.dimensions[ : nd_topological]
        index_tuple     = np.unravel_index(input_index, self.input_sdr.dimensions)
        topo_index      = index_tuple[ : nd_topological]
        return np.array(topo_index)

    def output_location(self, output_index):
        """
        Argument output_index is flat_index of postsynaptic output site.

        Returns the location of the given output in the input space.
        """
        nd_topological  = len(self.radii)
        topo_out_dims   = self.output_sdr.dimensions[ : nd_topological]
        index_tuple     = np.unravel_index(output_index, self.output_sdr.dimensions)
        topo_index      = index_tuple[ : nd_topological]
        topo_flat_index = np.ravel_multi_index(topo_index, topo_out_dims)
        location        = self.postsynaptic_locations[:, topo_flat_index]
        return location

    def measure_potential_pool_densities(self,):
        """
        This measures the fraction of inputs which potentially connect to an
        output, looking within the first three standard deviations of the
        outputs receptive field radius.  The areas are non-overlapping.  The
        results are averaged over all output sites.

        Returns triple of (potential_pool_density_1, potential_pool_density_2,
            potential_pool_density_3)
        """
        # Split the input space into topological and extra dimensions.
        radii            = np.array(self.radii)
        topo_dimensions  = self.input_sdr.dimensions[: len(radii)]
        extra_dimensions = self.input_sdr.dimensions[len(radii) :]

        # Density Statistics
        potential_pool_density_1 = 0
        potential_pool_density_2 = 0
        potential_pool_density_3 = 0

        # Determine the number of inputs in each area.
        extra_area   = np.product(extra_dimensions)
        num_inputs_1 = extra_area * math.pi * np.product(radii)
        num_inputs_2 = extra_area * math.pi * np.product(2 * radii)
        num_inputs_3 = extra_area * math.pi * np.product(3 * radii)
        num_inputs_3 -= num_inputs_2
        num_inputs_2 -= num_inputs_1

        for out_idx in range(self.output_sdr.size):
            center         = self.output_location(out_idx)
            potential_size = self.postsynaptic_sources_sizes[out_idx]
            potential_pool = self.postsynaptic_sources[out_idx][: potential_size]
            pp_locations   = [self.input_location(idx) for idx in potential_pool]
            if not pp_locations:
                continue  # No potential synapses, let density == 0
            displacements  = pp_locations - center

            # Measure in terms of standard deviations of their distribution.
            deviations = displacements / radii
            distances  = np.sum(deviations**2, axis=1)**.5
            # Count how many inputs fall within 1,2,3 radii of the output location.
            pp_size_1  = np.count_nonzero(distances <= 1)
            pp_size_2  = np.count_nonzero(np.logical_and(distances > 1, distances <= 2))
            pp_size_3  = np.count_nonzero(np.logical_and(distances > 2, distances <= 3))
            potential_pool_density_1 += pp_size_1 / num_inputs_1
            potential_pool_density_2 += pp_size_2 / num_inputs_2
            potential_pool_density_3 += pp_size_3 / num_inputs_3

        potential_pool_density_1 = potential_pool_density_1 / self.output_sdr.size
        potential_pool_density_2 = potential_pool_density_2 / self.output_sdr.size
        potential_pool_density_3 = potential_pool_density_3 / self.output_sdr.size

        return (potential_pool_density_1, potential_pool_density_2, potential_pool_density_3,)

    def statistics(self):
        stats = 'Synapse Manager Statistics:\n'

        # Build a small table of min/mean/std/max for pre & post, potential &
        # connected synapse counts.
        postsyn_potential = self.postsynaptic_sources_sizes
        postsyn_connected = []
        threshold         = PERMANENCE(self.permanence_thresh)
        for pp_perm, pp_sz in zip(self.postsynaptic_permanences, postsyn_potential):
            postsyn_connected.append(np.count_nonzero(pp_perm[ : pp_sz] >= threshold))
        entries = [
            ("Potential Synapses per Input ", self.presynaptic_sinks_sizes),
            ("Connected Synapses per Input ", self.presynaptic_partitions),
        ]
        unpopulated_outputs_exist = any(pp_size == 0 for pp_size in postsyn_potential)
        if not unpopulated_outputs_exist:
            entries.extend((
                ("Potential Synapses per Output", postsyn_potential),
                ("Connected Synapses per Output", postsyn_connected),))
        if unpopulated_outputs_exist:
            potentl_per_used_output = np.compress(postsyn_potential, postsyn_potential)
            connect_per_used_output = np.compress(postsyn_potential, postsyn_connected)
            if len(connect_per_used_output) > 0:
                entries.extend((
                    ("Pot Syns per Populated Output", potentl_per_used_output),
                    ("Con Syns per Populated Output", connect_per_used_output),))

        header  = ' '*len(entries[0][0])
        header += ''.join([' | %5s'%c for c in ['min', 'mean','std', 'max']]) + '\n'
        stats  += header
        for name, data in entries:
            columns = ( name,
                        int(round(np.min(data))),
                        int(round(np.mean(data))),
                        int(round(np.std(data))),
                        int(round(np.max(data))),)
            stats += '{} | {: >5d} | {: >5d} | {: >5d} | {: >5d}\n'.format(*columns)

        if unpopulated_outputs_exist:
            populated_outputs = np.count_nonzero(postsyn_potential)
            stats += "Populated Outputs: %g%%\n"%(100 * populated_outputs / self.output_sdr.size)

        if self.radii is not None and len(self.radii):
            pp_density_1, pp_density_2, pp_density_3 = self.measure_potential_pool_densities()
            stats += 'Potential Pool Density 1/2/3: {:5g} / {:5g} / {:5g}\n'.format(
                pp_density_1, pp_density_2, pp_density_3,)

            pp_mask = [np.count_nonzero(area) for area in self.postsynaptic_source_distributions]
            max_pp  = np.max(pp_mask)
            avg_pp  = np.mean(pp_mask)
            min_pp  = np.min(pp_mask)
            stats += 'Maximum Potential Pool Size Max/Avg/Min %d / %d / %d\n'%(max_pp, int(round(avg_pp)), min_pp,)

        return stats

    def check_data_integrity(self):
        """
        This method checks that the internal data structures are all okay.

        Checks that all data tables exist and have the correct data type.
        Checks that input -> output and output -> input yields the same connection.
        Checks that presyn_parts is okay.
        Checks that presyn_sizes is okay.
        Checks that postsyn_connected_counts is okay.
        Checks that all perms in range [0, 1]
        Checks that presyn-perms == postsyn-perms
        """
        presyn_perms = getattr(self, "presynaptic_permanences", None)
        # Check postsynaptic/output side data tables.
        for out_idx1 in range(self.output_sdr.size):
            sources1_inner = self.postsynaptic_sources[out_idx1]
            sources2_inner = self.postsynaptic_source_side_index[out_idx1]
            permanences    = self.postsynaptic_permanences[out_idx1]
            sources_size   = self.postsynaptic_sources_sizes[out_idx1]
            con_count      = self.postsynaptic_connected_count[out_idx1]
            # Check table size and type.
            assert(len(sources1_inner.shape) == 1)
            assert(sources1_inner.shape == sources2_inner.shape)
            assert(sources1_inner.shape == permanences.shape)
            assert(sources_size <= sources1_inner.shape[0])
            assert(sources1_inner.dtype == INDEX)
            assert(sources2_inner.dtype == INDEX)
            assert(permanences.dtype == PERMANENCE)
            assert(np.all(permanences[ : sources_size] >= 0.))
            assert(np.all(permanences[ : sources_size] <= 1.))
            assert(con_count == np.count_nonzero(permanences >= self.permanence_thresh))
            # Check Output -> Input linkage.
            for out_idx2 in range(sources_size):
                inp_idx1 = sources1_inner[out_idx2]
                inp_idx2 = sources2_inner[out_idx2]
                assert(inp_idx2 < self.presynaptic_sinks_sizes[inp_idx1])

                sinks1_inner = self.presynaptic_sinks[inp_idx1]
                sinks2_inner = self.presynaptic_sink_side_index[inp_idx1]
                assert(sinks1_inner[inp_idx2] == out_idx1)
                assert(sinks2_inner[inp_idx2] == out_idx2)
                if presyn_perms is not None:
                    assert(presyn_perms[inp_idx1][inp_idx2] == permanences[out_idx2])

        # Check presynaptic/input side data tables.
        assert(self.presynaptic_sinks_sizes.shape == (self.input_sdr.size,))
        assert(self.presynaptic_partitions.shape  == (self.input_sdr.size,))
        assert(self.presynaptic_sinks_sizes.dtype == INDEX)
        assert(self.presynaptic_partitions.dtype  == INDEX)
        for inp_idx1 in range(self.input_sdr.size):
            sinks1_inner = self.presynaptic_sinks[inp_idx1]
            sinks2_inner = self.presynaptic_sink_side_index[inp_idx1]
            # Check table size and type.
            assert(len(sinks1_inner.shape) == 1)
            assert(sinks1_inner.shape == sinks2_inner.shape)
            assert(sinks1_inner.dtype == INDEX)
            assert(sinks2_inner.dtype == INDEX)

            if presyn_perms is not None:
                presyn_perms_inner = presyn_perms[inp_idx1]
                assert(sinks1_inner.shape == presyn_perms_inner.shape)
                assert(presyn_perms_inner.dtype == PERMANENCE)

            sinks_size = self.presynaptic_sinks_sizes[inp_idx1]
            assert(sinks_size <= sinks1_inner.shape[0])

            partition = self.presynaptic_partitions[inp_idx1]
            assert(partition >= 0 and partition <= sinks_size)

            # Check Input -> Output linkage.
            for inp_idx2 in range(sinks_size):
                out_idx1 = sinks1_inner[inp_idx2]
                out_idx2 = sinks2_inner[inp_idx2]
                assert(out_idx2 < self.postsynaptic_sources_sizes[out_idx1])

                sources1_inner = self.postsynaptic_sources[out_idx1]
                sources2_inner = self.postsynaptic_source_side_index[out_idx1]
                assert(sources1_inner[out_idx2] == inp_idx1)
                assert(sources2_inner[out_idx2] == inp_idx2)
                permanence = self.postsynaptic_permanences[out_idx1][out_idx2]
                if presyn_perms is not None:
                    assert(presyn_perms_inner[inp_idx2] == permanence)
                # Check partition
                if permanence >= self.permanence_thresh:
                    assert(inp_idx2 < partition)
                else:
                    assert(inp_idx2 >= partition)


@cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
@cython.wraparound(False)  # Turns off negative index wrapping for entire function.
@cython.profile(DEBUG)
cdef _swap_inputs(
    np.ndarray[INDEX_t] sinks_inner,
    np.ndarray[INDEX_t] sinks2_inner,
    np.ndarray[PERMANENCE_t] presyn_perms_inner,
    np.ndarray[object]  sources2,
    int inp_idx2_a,
    int inp_idx2_b):
    """
    Swap the positions of two synapses in the presynaptic/input side tables.
    """
    cdef:
        np.ndarray[INDEX_t] sources2_inner
        int out_idx1_a, out_idx2_a
        int out_idx1_b, out_idx2_b
        PERMANENCE_t permanence_swap

    # Get the output indices of both inputs.
    out_idx1_a = sinks_inner[inp_idx2_a]
    out_idx2_a = sinks2_inner[inp_idx2_a]
    out_idx1_b = sinks_inner[inp_idx2_b]
    out_idx2_b = sinks2_inner[inp_idx2_b]

    # Swap in all input side tables.
    sinks_inner[inp_idx2_a] = out_idx1_b
    sinks_inner[inp_idx2_b] = out_idx1_a
    sinks2_inner[inp_idx2_a] = out_idx2_b
    sinks2_inner[inp_idx2_b] = out_idx2_a
    if presyn_perms_inner is not None:
        permanence_swap = presyn_perms_inner[inp_idx2_a]
        presyn_perms_inner[inp_idx2_a] = presyn_perms_inner[inp_idx2_b]
        presyn_perms_inner[inp_idx2_b] = permanence_swap

    # Notify the output side of the synapses new locations in the input tables.
    sources2_inner = <np.ndarray[INDEX_t]> sources2[out_idx1_a]
    sources2_inner[out_idx2_a] = inp_idx2_b
    sources2_inner = <np.ndarray[INDEX_t]> sources2[out_idx1_b]
    sources2_inner[out_idx2_b] = inp_idx2_a

@cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
@cython.wraparound(False)  # Turns off negative index wrapping for entire function.
@cython.profile(DEBUG)
cdef _swap_outputs(
    np.ndarray[INDEX_t] sources_inner,
    np.ndarray[INDEX_t] sources2_inner,
    np.ndarray[PERMANENCE_t] perms_inner,
    np.ndarray[object]  sinks2,
    int out_idx2_a,
    int out_idx2_b,):
    """
    Swap two synapses in the postsynaptic/output side tables.
    """
    cdef:
        int inp_idx1_a, inp_idx2_a
        int inp_idx1_b, inp_idx2_b
        PERMANENCE_t perm_swap
        np.ndarray[INDEX_t] sinks2_inner

    # Get input indices of both outputs.
    inp_idx1_a = sources_inner[out_idx2_a]
    inp_idx2_a = sources2_inner[out_idx2_a]
    inp_idx1_b = sources_inner[out_idx2_b]
    inp_idx2_b = sources2_inner[out_idx2_b]

    # Swap in all the output side tables.
    sources_inner[out_idx2_a]  = inp_idx1_b
    sources_inner[out_idx2_b]  = inp_idx1_a
    sources2_inner[out_idx2_a] = inp_idx2_b
    sources2_inner[out_idx2_b] = inp_idx2_a
    perm_swap               = perms_inner[out_idx2_a]
    perms_inner[out_idx2_a] = perms_inner[out_idx2_b]
    perms_inner[out_idx2_b] = perm_swap

    # Notify the input side tables of the synapses new locations in the output
    # side tables.
    sinks2_inner = <np.ndarray[INDEX_t]> sinks2[inp_idx1_a]
    sinks2_inner[inp_idx2_a] = out_idx2_b
    sinks2_inner = <np.ndarray[INDEX_t]> sinks2[inp_idx1_b]
    sinks2_inner[inp_idx2_b] = out_idx2_a

@cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
@cython.wraparound(False)  # Turns off negative index wrapping for entire function.
@cython.profile(DEBUG)
cdef _find_min_perm_synapses(
    int                      n_synapses,
    np.ndarray[INDEX_t]      sources_inner,
    np.ndarray[PERMANENCE_t] permanences_inner,
    INDEX_t                  sources_size,
    np.ndarray[INDEX_t]      exclude_sources,):
    """
    Returns an array of indicies into the given sources list.
    """
    cdef:
        int n_candidates
        np.ndarray[INDEX_t]      candidate_out_idx2
        np.ndarray[PERMANENCE_t] candidate_permanences

    sources_inner     = sources_inner[ : sources_size]
    permanences_inner = permanences_inner[ : sources_size]

    candidate_out_idx2 = np.arange(sources_size, dtype=INDEX)

    include_sources_mask = np.in1d(
        sources_inner,
        exclude_sources,
        invert        = True,
        assume_unique = True,)

    candidate_out_idx2    = candidate_out_idx2[include_sources_mask]
    candidate_permanences = permanences_inner[include_sources_mask]
    if candidate_permanences.shape[0] == 0:
        return np.empty(0, dtype=INDEX)

    n_candidates    = _min(n_synapses, candidate_permanences.shape[0])
    candidate_index = np.argpartition(candidate_permanences, n_candidates-1)
    candidate_index = candidate_index[: n_synapses]
    return candidate_out_idx2[candidate_index]

@cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
@cython.wraparound(False)  # Turns off negative index wrapping for entire function.
@cython.profile(DEBUG)
cdef _remove_synapse(
    np.ndarray[object]  sources,
    np.ndarray[object]  sources2,
    np.ndarray[object]  perms,
    np.ndarray[INDEX_t] sources_sizes,
    np.ndarray[object]  sinks,
    np.ndarray[object]  sinks2,
    np.ndarray[object]  presyn_perms,
    np.ndarray[INDEX_t] sink_parts,
    np.ndarray[INDEX_t] sink_sizes,
    PERMANENCE_t        permanence_thresh,
    int                 out_idx1,
    int                 out_idx2,):
    cdef:
        # Inner data tables.
        np.ndarray[INDEX_t] sources_inner    = sources[out_idx1]
        np.ndarray[INDEX_t] sources2_inner   = sources2[out_idx1]
        np.ndarray[PERMANENCE_t] perms_inner = perms[out_idx1]

    cdef:
        # Info about synapse to be removed.
        INDEX_t                  inp_idx1           = sources_inner[out_idx2]
        INDEX_t                  inp_idx2           = sources2_inner[out_idx2]
        PERMANENCE_t             perm_value         = perms_inner[out_idx2]
        np.ndarray[INDEX_t]      sinks_inner        = sinks[inp_idx1]
        np.ndarray[INDEX_t]      sinks2_inner       = sinks2[inp_idx1]
        np.ndarray[PERMANENCE_t] presyn_perms_inner = None
    if presyn_perms is not None:
        presyn_perms_inner = presyn_perms[inp_idx1]

    cdef:
        # Local variables.
        int inp_idx2_swap
        int out_idx2_swap

    # Swap the input to end of connected partition.
    if perm_value >= permanence_thresh:
        inp_idx2_swap = sink_parts[inp_idx1] - 1
        _swap_inputs(
            sinks_inner,
            sinks2_inner,
            presyn_perms_inner,
            sources2,
            inp_idx2,
            inp_idx2_swap,)
        inp_idx2 = inp_idx2_swap    # Update location of synapse to be removed.
        # Decrement the partition since this synapse is no longer connected.
        sink_parts[inp_idx1] -= 1

    # Swap the input to the end of the input data tables.
    inp_idx2_swap = sink_sizes[inp_idx1] - 1
    _swap_inputs(
        sinks_inner,
        sinks2_inner,
        presyn_perms_inner,
        sources2,
        inp_idx2,
        inp_idx2_swap,)
    sink_sizes[inp_idx1] -= 1

    # Swap to end of output data tables and decrement the data tables logical size.
    out_idx2_swap = sources_sizes[out_idx1] - 1
    _swap_outputs(
        sources_inner,
        sources2_inner,
        perms_inner,
        sinks2,
        out_idx2,
        out_idx2_swap,)
    sources_sizes[out_idx1] -= 1

def _setup_topology(input_dimensions, output_dimensions, radii,):
    """
    Returns pair (output_locations, output_source_distributions)

    output_locations[topo-dimension][output-topo-index] = Location in input
    space.  This does not hold extra dimensions, only topological ones.  The
    outputs are indexed by their topological location's flat index which is
    different from the normal index (which includes their extra dimensions).

    output_source_distributions contains the probability density function for
    forming connections.  It is an array of arrays. The outer array contains an
    entry for each output location. The inner array contains the likelyhood of
    forming a potential synapse from each input location.   Extra dimensions are
    omitted from both the inputs and outputs.
    output_source_distributions[topological_output_index] = PDF
    PDF[topological_input_index] = probability.
    """
    radii = np.array(radii)

    # Split the input space into topological and extra dimensions.
    nd_topological    = len(radii)
    inp_topo_dims     = input_dimensions[ : nd_topological]
    out_topo_dims     = output_dimensions[ : nd_topological]
    out_topo_dim_area = np.product(out_topo_dims)

    # Find where the outputs are in the input space.
    output_ranges     = [slice(0, size) for size in out_topo_dims]
    output_index      = np.mgrid[output_ranges]
    output_locations  = [dim.flatten() for dim in output_index]
    padding           = radii
    inp_topo_max_idx  = np.subtract(inp_topo_dims, 1)
    scale_outputs_to  = np.subtract(inp_topo_max_idx, np.multiply(2, padding))
    out_topo_max_idx  = np.subtract(out_topo_dims, 1)
    output_spacing    = np.divide(scale_outputs_to, out_topo_max_idx)
    output_spacing    = np.nan_to_num(output_spacing)
    output_locations *= output_spacing.reshape(nd_topological, 1)
    output_locations += np.array(padding).reshape(nd_topological, 1)

    output_source_distributions = np.empty(out_topo_dim_area, dtype=object)
    # Fill in the output_source_distributions table, by iterating through every
    # topological area in the output space.
    for out_topo_idx in range(out_topo_dim_area):
        location = output_locations[:, out_topo_idx]

        # Find the distribution of inputs to this output location one dimension
        # at a time.  This works because gaussian distributions are linearly
        # separable and can be combined with the outer product.
        multivariate_src_dist = None
        for loc, radius, inp_dim_size in zip(location, radii, inp_topo_dims):
            src_dist = scipy.stats.norm(loc, radius)
            src_dist = src_dist.pdf(np.arange(inp_dim_size))
            # Combine the source-distribution for this dimension with that of
            # all of the previous dimensions.
            if multivariate_src_dist is None:
                multivariate_src_dist = src_dist
            else:
                multivariate_src_dist = np.outer(multivariate_src_dist, src_dist)

        # Normalize such that the sum is 1.
        multivariate_src_dist /= np.sum(multivariate_src_dist)
        # Filter out inputs which have a very low probability of activating.
        remove_potential_connections = multivariate_src_dist < 1e-5
        multivariate_src_dist[remove_potential_connections] = 0.

        output_source_distributions[out_topo_idx] = multivariate_src_dist
    return output_locations, output_source_distributions
