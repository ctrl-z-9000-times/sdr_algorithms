# Written by David McDougall, 2018
# cython: language_level=3

DEF DEBUG = False
debug = DEBUG

import numpy as np
cimport numpy as np
cimport cython
import math
import copy
import random
from sdr import SDR

ctypedef np.float32_t PERMANENCE_t  # For Cython's compile time type system.
PERMANENCE = np.float32             # For Python3's run time type system.

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

    Attributes presynaptic_sinks, presynaptic_sinks_sizes
        presynaptic_sinks[input-index] = 1D array of potential output indices.
        The sinks table is associates each input (presynapse) with its outputs
        (postsynapses), allowing for fast feed forward calculations.  
        presynaptic_sinks_sizes[input-index] = The logical size of the
        presynaptic_sinks entry.  The presynaptic_sinks are  allocated with
        extra space to facilitate adding and removing synapses.

    Attribute presynaptic_partitions
        presynapic_partitions[input-index] = Number of synapses this input makes.
        The presynaptic_sinks table is partitioned by permanence value such that
        the first synapses in the table are connected and the remaining are
        disconnected.

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
        permanence_thresh = 0,
        permanence_inc    = 0,
        permanence_dec    = 0,
        weighted          = False,):
        """
        Argument input_sdr is the presynaptic input activations.
        Argument output_sdr is the postsynaptic segment sites.

        Argument init_dist is (mean, std) of permanence value random initial distribution.
        Optional Argument permanence_thresh ...
        Optional Argument permanence_inc ...
        Optional Argument permanence_dec ...

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

        self.postsynaptic_sources       = np.empty(self.output_sdr.size, dtype=object)
        self.postsynaptic_permanences   = np.empty(self.output_sdr.size, dtype=object)
        for idx in range(self.output_sdr.size):
            self.postsynaptic_sources[idx]     = np.empty((0,), dtype=INDEX)
            self.postsynaptic_permanences[idx] = np.empty((0,), dtype=PERMANENCE)
        self.rebuild_indexes()  # Initializes sinks index and the rest.

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
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
            np.ndarray[dtype=object] sinks_table = self.presynaptic_sinks
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
                potential_x[out_idx] += inp_value
            # Tally the potential synapses.
            for inp_idx2 in range(partition, sink_sizes[inp_idx]):
                out_idx = sinks_inner[inp_idx2]
                potential_x[out_idx] += inp_value

        return (excitement.reshape(self.output_sdr.dimensions),
                potential_x.reshape(self.output_sdr.dimensions))

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
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

    # TODO: I've reason to believe that the NZ values should not be used by
    # hebbian learning.  Sort this out and remove the extra multiplications if
    # applicable.  NZ values is only useful for feed forward computation.
    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def learn(self, input_sdr=None, output_sdr=None,
        permanence_inc = None,
        permanence_dec = None,):
        """
        Update the permanences of active outputs using the most recently given
        inputs.

        Argument output_sdr contains the indices of the outputs to apply hebbian
            learning to.

        Argument input_sdr ...

        Optional Arguments permanence_inc and permanence_dec take precedence
           over any value passed to this classes initialize method.
        """
        self.input_sdr.assign(input_sdr)
        self.output_sdr.assign(output_sdr)
        cdef:
            # Data tables
            np.ndarray[object]      sources          = self.postsynaptic_sources
            np.ndarray[object]      sources2         = self.postsynaptic_source_side_index
            np.ndarray[object]      permanences      = self.postsynaptic_permanences
            np.ndarray[object]      sinks            = self.presynaptic_sinks
            np.ndarray[object]      sinks2           = self.presynaptic_sink_side_index
            np.ndarray[object]      presyn_perms     = getattr(self, 'presynaptic_permanences', None)
            np.ndarray[INDEX_t]     sinks_sizes      = self.presynaptic_sinks_sizes
            np.ndarray[INDEX_t]     sinks_parts      = self.presynaptic_partitions
            np.ndarray[np.int_t]    output_activity  = self.output_sdr.flat_index
            np.ndarray[np.uint8_t]  output_nz_values = self.output_sdr.nz_values
            np.ndarray[np.uint8_t]  input_activity   = self.input_sdr.dense.reshape(-1)

            # Inner array pointers
            np.ndarray[INDEX_t]      sources_inner
            np.ndarray[INDEX_t]      sources2_inner
            np.ndarray[INDEX_t]      sources2_inner_move
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[INDEX_t]      sinks_inner
            np.ndarray[INDEX_t]      sinks2_inner
            np.ndarray[PERMANENCE_t] presyn_perms_inner

            # Indexes and locals
            INDEX_t out_iter, out_idx1, out_idx2, out_idx1_swap, out_idx2_swap
            INDEX_t inp_idx1, inp_idx2, inp_idx2_swap
            INDEX_t sink_size
            INDEX_t partition
            np.uint8_t input_value, output_value
            PERMANENCE_t perm_value
            PERMANENCE_t base_inc, base_dec
            PERMANENCE_t inc, dec, thresh = self.permanence_thresh
            PERMANENCE_t perm_swap
            bint syn_prior, syn_post
            np.ndarray[INDEX_t] buffer = np.full(20, -1, dtype=INDEX)

        # Arguments override initialized or default values.
        base_inc = permanence_inc if permanence_inc is not None else self.permanence_inc
        base_dec = permanence_dec if permanence_dec is not None else self.permanence_dec
        if base_inc == 0. and base_dec == 0.:
            return

        for out_iter in range(output_activity.shape[0]):
            out_idx1 = output_activity[out_iter]

            # <type?> is safe case, <type> is unsafe cast.
            sources_inner    = <np.ndarray[INDEX_t]>      sources[out_idx1]
            sources2_inner   = <np.ndarray[INDEX_t]>      sources2[out_idx1]
            perms_inner      = <np.ndarray[PERMANENCE_t]> permanences[out_idx1]

            # Scale permanence updates by the activation strength.
            output_value = output_nz_values[out_iter]
            inc = base_inc * output_value
            dec = base_dec * output_value

            for out_idx2 in range(sources_inner.shape[0]):
                inp_idx1     = sources_inner[out_idx2]
                perm_value   = perms_inner[out_idx2]
                syn_prior    = perm_value >= thresh
                # Hebbian Learning.
                input_value  = input_activity[inp_idx1]
                if input_value != 0:
                    perm_value += inc * input_value
                else:
                    perm_value -= dec
                # Clip to [0, 1]
                if perm_value > 1.:
                    perm_value = 1.
                elif perm_value < 0.:
                    perm_value = 0.

                syn_post      = perm_value >= thresh
                perms_inner[out_idx2] = perm_value

                if presyn_perms is not None:
                    inp_idx2     = sources2_inner[out_idx2]
                    presyn_perms_inner = <np.ndarray[PERMANENCE_t]> presyn_perms[inp_idx1]
                    presyn_perms_inner[inp_idx2] = perm_value

                if syn_prior != syn_post:
                    inp_idx2     = sources2_inner[out_idx2]
                    sinks_inner  = <np.ndarray[INDEX_t]> sinks[inp_idx1]
                    sinks2_inner = <np.ndarray[INDEX_t]> sinks2[inp_idx1]
                    partition    = sinks_parts[inp_idx1]

                    if syn_post:
                        # Swap this synapse to the start of the disconnected
                        # segment and increment the partition over it.
                        inp_idx2_swap = partition
                        sinks_parts[inp_idx1] += 1
                    else:
                        # Swap this synapse to the end of the connected segment
                        # and decrement the partition over of.
                        inp_idx2_swap = partition - 1
                        sinks_parts[inp_idx1] -= 1

                    # Swap the inputs.  Get the address of the evicted synapse
                    # so that it can be relocated.
                    out_idx1_swap = sinks_inner[inp_idx2_swap]
                    out_idx2_swap = sinks2_inner[inp_idx2_swap]
                    # Move the synapse in the sinks tables, overwriting the
                    # synapse at [inp_idx, inp_idx2_swap, out_idx1_swap, out_idx2_swap]
                    sinks_inner[inp_idx2_swap]  = out_idx1
                    sinks2_inner[inp_idx2_swap] = out_idx2
                    sources2_inner[out_idx2]    = inp_idx2_swap

                    # Relocate the evicted synapse.
                    sinks_inner[inp_idx2]       = out_idx1_swap
                    sinks2_inner[inp_idx2]      = out_idx2_swap
                    # Notify the postsynaptic side of the synapse which got moved.
                    sources2_inner_move = <np.ndarray[INDEX_t]> sources2[out_idx1_swap]
                    sources2_inner_move[out_idx2_swap] = inp_idx2

                    # Simple swap for the presynaptic permanence table.
                    if presyn_perms is not None:
                        perm_swap = presyn_perms_inner[inp_idx2_swap]
                        presyn_perms_inner[inp_idx2_swap] = presyn_perms_inner[inp_idx2]
                        presyn_perms_inner[inp_idx2] = perm_swap

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    @cython.profile(DEBUG)
    def add_synapses(self, maximum_synapses, maximum_new_synapses,
        init_dist  = None,
        input_sdr  = None,
        output_sdr = None,):
        """
        Sources for the new synapses are sampled from the input_sdr.

        Argument input_sdr contains the sources to grow new synapses from.  It
            is a probability density function, this SDR's values are
            proportional to the likelyhood that each input will be sampled.

        Argument output_sdr contains the the outputs to add synapses to.  The
            values stored in the output_sdr are the number of new synapses to
            add to each output.

        Argument maximum_synapses which a segment can have.

        Argument init_dist is (mean, std) of synapse initial permanence
        distribution.
        """
        # Arguments.
        self.input_sdr.assign(input_sdr)
        self.output_sdr.assign(output_sdr)
        if len(self.input_sdr) == 0 or len(self.output_sdr) == 0:
            return
        cdef:
            np.ndarray[INDEX_t]     inputs    = np.array(self.input_sdr.flat_index, dtype=INDEX)
            np.ndarray[np.double_t] nz_values = self.input_sdr.nz_values / np.sum(self.input_sdr.nz_values)
            np.ndarray[np.long_t]   outputs   = self.output_sdr.flat_index
            np.ndarray[np.uint8_t]  new_syns  = self.output_sdr.nz_values
        if init_dist is not None:
            init_mean, init_std = init_dist
        else:
            init_mean, init_std = self.init_dist

        cdef:
            # Data Tables.
            np.ndarray[object] sources       = self.postsynaptic_sources
            np.ndarray[object] sources2      = self.postsynaptic_source_side_index
            np.ndarray[object] permanences   = self.postsynaptic_permanences
            np.ndarray[object] presyn_perms  = getattr(self, 'presynaptic_permanences', None)
            np.ndarray[object] sinks         = self.presynaptic_sinks
            np.ndarray[object] sinks2        = self.presynaptic_sink_side_index
            np.ndarray[INDEX_t] sinks_sizes  = self.presynaptic_sinks_sizes

            # Inner array pointers.
            np.ndarray[INDEX_t]      sources_inner
            np.ndarray[INDEX_t]      sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[INDEX_t]      sinks_inner
            np.ndarray[INDEX_t]      sinks2_inner
            np.ndarray[PERMANENCE_t] presyn_perms_inner

            # Locals indexes and counters.
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
            int orig_nsyn

            np.ndarray[INDEX_t]      index_buffer = np.full(max_new_syn, -1,           dtype=INDEX)
            np.ndarray[PERMANENCE_t] perms_buffer = np.full(max_new_syn, float('nan'), dtype=PERMANENCE)

        for out_iter in range(outputs.shape[0]):
            out_idx           = outputs[out_iter]
            nsyn_to_add       = new_syns[out_iter]
            sources_inner     = <np.ndarray[INDEX_t]>  sources[out_idx] # Unsafe Typecasts!
            sources2_inner    = <np.ndarray[INDEX_t]>  sources2[out_idx]
            perms_inner       = <np.ndarray[PERMANENCE_t]> permanences[out_idx]

            # Enforce the maximum number of synapses per output.
            nsyn_to_add = _min(nsyn_to_add, max_syn - sources_inner.shape[0])
            # Can not add more synapses than there are available inputs for.
            nsyn_to_add = _min(nsyn_to_add, max_new_syn)

            # Randomly select inputs.
            candidate_sources = np.random.choice(inputs,
                                    size    = nsyn_to_add,
                                    replace = False,
                                    p       = nz_values,)
            # Filter out inputs which have already been connected to.
            unique_sources    = np.isin(candidate_sources, sources_inner, invert=True,
                                        assume_unique = True,)
            new_sources = candidate_sources[unique_sources]

            # Random initial distribution of permanence values.
            new_permanances_f64 = np.random.normal(init_mean, init_std, size=new_sources.shape[0])
            new_permanances     = np.array(new_permanances_f64, dtype=PERMANENCE)
            np.clip(new_permanances, 0., 1., out=new_permanances)

            # Append to the postsynaptic source tables.
            orig_nsyn            = sources_inner.shape[0]
            sources[out_idx]     = sources_inner  = np.append(sources_inner, new_sources)
            permanences[out_idx] = perms_inner    = np.append(perms_inner, new_permanances)
            sources2[out_idx]    = sources2_inner = np.append(sources2_inner, index_buffer[ : new_sources.shape[0]])

            # Process each new synapse into the data tables.
            for out_idx2 in range(orig_nsyn, sources_inner.shape[0]):
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

                # Swap synapses to maintain partitioned order.
                if perm >= thresh:
                    1/0

                    # Gather presynaptic sinks tables.
                    inp = sources_inner[out_idx2]   # 'inp' should really be named 'inp_idx1'.
                    if presyn_perms is not None:
                        presyn_perms_inner = presyn_perms[inp]
                    # Append new synapse to sinks table.
                    inp_idx2         = sinks_sizes[inp]
                    sources2_inner[out_idx2] = inp_idx2 # Link sources back to sinks.
                    sinks_sizes[inp] += 1

    @cython.profile(DEBUG)
    def normally_distributed_connections(self, potential_pool, radii):
        """
        Makes synapses from inputs to outputs within their local neighborhood.
        The outputs exist on a uniform grid which is stretched over the input
        space.  An outputs local neighborhood is defined by a gaussian window
        centered over the output with standard deviations given by argument
        radii.

        Argument potential_pool is the number of potential inputs to connect
                 each output to.

        Argument radii is tuple of standard deivations. Radii units are the
                 input space units.  Radii defines the topology of the
                 connections.  If radii are shorter than the number of input or
                 output space dimensions then the trailing input dimensions are
                 not considered topological dimensions.  These 'extra'
                 dimensions are treated with uniform probability; only distances
                 in the topological dimensions effect the probability of forming
                 a potential synapse.

        Attributes set by this method:
            self.potential_pool_density_1,
            self.potential_pool_density_2,
            self.potential_pool_density_3,
                These measure the average fraction of inputs which are
                potentially connected to each outputs, looking within the first
                three standard deviations of the outputs receptive field.  The
                areas are non-overlapping.  These are incorperated into the
                statistics method if they are available.

        Returns inhibition_radii which is the is the radii after converting it
                into output space units.

        Note: this method does NOT check for duplicate synapses and should only
        be called on an empty synapse manager with no existing synapses.
        """
        radii = np.array(self.radii)
        if len(radii) == 0:
            return self.uniformly_distributed_connections(potential_pool)
        assert(len(radii.shape) == 1)
        potential_pool      = int(round(potential_pool))
        init_mean, init_std = self.init_dist

        # Split the input space into topological and extra dimensions.
        topo_dimensions  = self.input_sdr.dimensions[: len(radii)]
        extra_dimensions = self.input_sdr.dimensions[len(radii) :]
        topo_output_dims = self.output_sdr.dimensions[: len(radii)]

        # Density Statistics
        potential_pool_density_1 = 0
        potential_pool_density_2 = 0
        potential_pool_density_3 = 0
        extra_area   = np.product(extra_dimensions)
        num_inputs_1 = extra_area * math.pi * np.product(radii)
        num_inputs_2 = extra_area * math.pi * np.product(2 * radii)
        num_inputs_3 = extra_area * math.pi * np.product(3 * radii)
        num_inputs_2 -= num_inputs_1
        num_inputs_3 -= num_inputs_1 + num_inputs_2

        # Find where the columns are in the input.
        output_ranges     = [slice(0, size) for size in self.output_sdr.dimensions]
        output_index      = np.mgrid[output_ranges]
        # output_locations[input-dimension][output-index] = Location in input
        # space.  This does not hold extra dimensions, only topological ones.
        output_locations  = [dim.flatten() for dim in output_index[: len(radii)]]
        padding           = radii   # No wrapping.
        expand_to         = np.subtract(topo_dimensions, np.multiply(2, padding))
        column_spacing    = np.divide(expand_to, topo_output_dims).reshape(len(topo_dimensions), 1)
        output_locations *= column_spacing
        output_locations += np.array(padding).reshape(len(topo_dimensions), 1)
        inhibition_radii  = radii / np.squeeze(column_spacing)
        self.output_locations = output_locations

        for column_index in range(self.output_sdr.size):
            center = output_locations[:, column_index]
            # Make potential-pool many unique input locations.  This is an
            # iterative process: sample the normal distribution, reject
            # duplicates, repeat until done.  Working pool holds the
            # intermediate input-coordinates until it's filled and ready to be
            # spliced into self.postsynaptic_sources[column-index, :]
            working_pool  = np.empty((0, len(self.input_sdr.dimensions)), dtype=np.int)
            empty_sources = potential_pool  # How many samples to take.
            for attempt in range(10):
                # Sample points from the input space and cast to valid indices.
                # Take more samples than are needed B/C some will not be viable.
                topo_pool     = np.random.normal(center, radii, 
                                    size=(max(256, 2*empty_sources), len(radii)))
                topo_pool     = np.rint(topo_pool)   # Round towards center
                # Discard samples which fall outside of the input space.
                out_of_bounds = np.logical_or(topo_pool < 0, topo_pool >= topo_dimensions)
                out_of_bounds = np.any(out_of_bounds, axis=1)
                topo_pool     = topo_pool[np.logical_not(out_of_bounds)]
                extra_pool    = np.random.uniform(0, extra_dimensions,
                                size=(topo_pool.shape[0], len(extra_dimensions)))
                extra_pool    = np.floor(extra_pool) # Round down to stay in half open range [0, dim)
                # Combine topo & extra dimensions into input space coordinates.
                pool          = np.concatenate([topo_pool, extra_pool], axis=1)
                pool          = np.array(pool, dtype=np.int)
                # Add the points to the working pool.
                working_pool  = np.concatenate([working_pool, pool], axis=0)
                # Reject duplicates.
                working_pool  = np.unique(working_pool, axis=0)
                empty_sources = potential_pool - working_pool.shape[0]
                if empty_sources <= 0:
                    break
            else:
                if empty_sources > .10 * potential_pool:
                    raise ValueError("Not enough sources to fill potential pool, %d too many."%empty_sources)
                else:
                    pass
                    # print("Warning: Could not find enough unique inputs, allowing %d fewer inputs..."%empty_sources)
            working_pool = working_pool[:potential_pool, :] # Discard extra samples

            # Measure some statistics about input density.
            displacements = working_pool[:, :len(topo_dimensions)] - center
            # Measure in terms of standard deviations of their distribution.
            deviations = displacements / radii
            distances  = np.sum(deviations**2, axis=1)**.5
            pp_size_1  = np.count_nonzero(distances <= 1)
            pp_size_2  = np.count_nonzero(np.logical_and(distances > 1, distances <= 2))
            pp_size_3  = np.count_nonzero(np.logical_and(distances > 2, distances <= 3))
            potential_pool_density_1 += pp_size_1 / num_inputs_1
            potential_pool_density_2 += pp_size_2 / num_inputs_2
            potential_pool_density_3 += pp_size_3 / num_inputs_3

            # Generate random initial distribution of permanence values.
            initial_permanences = np.random.normal(init_mean, init_std, size=(len(working_pool),))
            initial_permanences = np.clip(initial_permanences, 0, 1)
            initial_permanences = np.array(initial_permanences, dtype=PERMANENCE)

            # Flatten and write to output array.
            working_pool = np.ravel_multi_index(working_pool.T, self.input_sdr.dimensions)
            working_pool = np.array(working_pool, dtype=INDEX)
            self.postsynaptic_sources[column_index]     = np.append(self.postsynaptic_sources[column_index], working_pool)
            self.postsynaptic_permanences[column_index] = np.append(self.postsynaptic_permanences[column_index], initial_permanences)

        self.rebuild_indexes()

        self.potential_pool_density_1 = potential_pool_density_1 / self.output_sdr.size
        self.potential_pool_density_2 = potential_pool_density_2 / self.output_sdr.size
        self.potential_pool_density_3 = potential_pool_density_3 / self.output_sdr.size
        self.inhibition_radii = inhibition_radii
        return inhibition_radii

    @cython.profile(DEBUG)
    def uniformly_distributed_connections(self, potential_pool, init_dist=None):
        """
        Connect every output to potential_pool inputs.
        Directly sets the sources and permanence arrays, no returned value.
        Will raise ValueError if potential_pool is invalid.

        Argument potential_pool is the number of potential inputs to connect
                 each output to.

        Note: this method does NOT check for duplicate synapses and should only
        be called on an empty synapse manager with no existing synapses.
        """
        potential_pool = int(round(potential_pool))
        if init_dist is not None:
            init_mean, init_std = init_dist
        else:
            init_mean, init_std = self.init_dist
        for out_idx in range(self.output_sdr.size):
            syn_src = np.random.choice(self.input_sdr.size, potential_pool, replace=False)
            syn_src = np.array(syn_src, dtype=INDEX)
            # Random initial distribtution of permanence value
            syn_prm = np.random.normal(init_mean, init_std, size=syn_src.shape)
            syn_prm = np.clip(syn_prm, 0, 1)
            syn_prm = np.array(syn_prm, dtype=PERMANENCE)
            # Append new sources and permanences to primary tables.
            self.postsynaptic_sources[out_idx]     = np.append(self.postsynaptic_sources[out_idx], syn_src)
            self.postsynaptic_permanences[out_idx] = np.append(self.postsynaptic_permanences[out_idx], syn_prm)
        self.rebuild_indexes()

    @cython.profile(DEBUG)
    def rebuild_indexes(self):
        """
        This method uses the postsynaptic_sources and postsynaptic_permanences
        tables to rebuild all of the other needed tables.
        """
        cdef:
            # Data tables
            np.ndarray[object]  sources          = self.postsynaptic_sources
            np.ndarray[object]  permanences      = self.postsynaptic_permanences
            np.ndarray[object]  sources2         = np.empty(self.output_sdr.size, dtype=object)
            np.ndarray[object]  sinks            = np.empty(self.input_sdr.size,  dtype=object)
            np.ndarray[object]  sinks2           = np.empty(self.input_sdr.size,  dtype=object)
            np.ndarray[INDEX_t] sinks_sizes      = np.zeros(self.input_sdr.size,  dtype=INDEX)
            np.ndarray[INDEX_t] sinks_parts      = np.zeros(self.input_sdr.size,  dtype=INDEX)
            np.ndarray[object]  sink_perms       = np.empty(self.input_sdr.size, dtype=object)
            # Inner array pointers
            np.ndarray[INDEX_t]  sources_inner
            np.ndarray[INDEX_t]  sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            # Locals
            int inp_idx, inp_idx2, out_idx, synapse_num
            PERMANENCE_t perm_val, thresh = self.permanence_thresh
        index_buffer = np.full(20, -1,           dtype=INDEX)
        perms_buffer = np.full(20, float('nan'), dtype=PERMANENCE)
        # Initialize the index tables with python lists for fast append.
        for inp_idx in range(self.input_sdr.size):
            sinks[inp_idx]      = []
            sinks2[inp_idx]     = []
            sink_perms[inp_idx] = []
        # Iterate through every synapse from the output side, build the input
        # side tables.
        for out_idx in range(self.output_sdr.size):
            sources_inner     = sources[out_idx]
            perms_inner       = permanences[out_idx]
            num_sources       = len(sources_inner)
            sources2[out_idx] = np.full(num_sources, -1, dtype=INDEX)
            for synapse_num in range(num_sources):
                inp_idx  = sources_inner[synapse_num]
                perm_val = perms_inner[synapse_num]
                assert(perm_val >= 0. and perm_val <= 1.)
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
        if self.weighted:
            self.presynaptic_permanences    = sink_perms

    # TODO: Write a function which prunes out zero permanence synapses.

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

    def statistics(self):
        stats = 'Synapse Manager Statistics:\n'

        # Build a small table of min/mean/std/max for pre & post, potential &
        # connected synapse counts (16 values total).
        postsyn_potential = [pp.shape[0] for pp in self.postsynaptic_sources]
        threshold         = PERMANENCE(self.permanence_thresh)
        postsyn_connected = [np.count_nonzero(pp_perm >= threshold)
                                for pp_perm in self.postsynaptic_permanences]
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

        if hasattr(self, 'potential_pool_density_1'):
            stats += 'Potential Pool Density 1/2/3: {:5g} / {:5g} / {:5g}\n'.format(
                self.potential_pool_density_1,
                self.potential_pool_density_2,
                self.potential_pool_density_3,)

        return stats
