# Written by David McDougall, 2018

import numpy as np
import math
from fractions import Fraction
import random
import itertools

def _choose(n, r):
    """Number of ways to choose 'r' unique elements from set of 'n' elements."""
    factorial = math.factorial
    return factorial(n) // factorial(r) // factorial(n-r)

def _binary_entropy(p):
    p_ = (1 - p)
    s  = -p*np.log2(p) -p_*np.log2(p_)
    return np.mean(np.nan_to_num(s))


class SparseDistributedRepresentation:
    """
    This class represents both the specification and the momentary value of a
    Sparse Distributed Representation.  Classes initialized with SDRs can expect
    the SDRs values to change.  An SDR represents the state of a group of
    neurons.  An SDR consists of a value (unsigned 8-bit) for each neuron.

    Attribute dimensions is a tuple of integers, read only.
    Attribute size is the total number of neurons in SDR, read only.

    The following four attributes hold the current value of the SDR, and are
    read/writable.  Values assigned to one attribute are converted to the other
    formats as needed.  Assigning to to 'dense', 'index', or 'flat_index' will
    over write the current SDR value, only the most recent assignment to these
    three attributes is kept.  Also, assigning to these attributes will clear
    the 'nz_values' so 'nz_values' must be assigned to AFTER assigning to either
    'index' or 'flat_index'.

    Attribute dense is an np.ndarray, shape=self.dimensions, dtype=np.uint8

    Attribute index is a tuple of np.ndarray, len(sdr.index) == len(sdr.dimensions)
              sdr.index[dim].shape=(num-active-bits), dtype=np.int

    Attribute flat_index is an np.ndarray, shape=(all-one-dim,), dtype=np.int
              The flat index is the index into the flattened SDR.

    Attribute nz_values is an np.ndarray, shape=(len(sdr),), dtype=np.uint8
              Contains all non-zero values in the SDR, in same order as 
              sdr.index and sdr.flat_index.  Defaults to all ones.
    """
    def __init__(self, specification,
        activation_frequency_alpha=None,
        average_overlap_alpha=None):
        """
        Argument specification can be either one of the following:
            A tuple of numbers, which declares the dimensions.
            An SDR, which makes this instance a shallow copy of it.

        Optional Argument activation_frequency_alpha.  If given, this SDR will
            automatically track the moving exponential average of each bits
            activation frequency.  The activation frequencies are updated every
            time this SDR is assigned to.  These records are NOT copied to any
            other SDRs, even by the copy constructor.  The given alpha is the
            weight given to each new activation.  Giving a False value
            (including None and 0) will disable this feature entirely, it is OFF
            by default.  The activation frequencies are available at attribute
            sdr.activation_frequency, dtype=np.float32, shape=(sdr.size,).

        Optional Argument average_overlap_alpha.  If given, this SDR will
            automatically track the moving exponential average of the overlap
            between sucessive values of this SDR and store them at the attribute
            sdr.average_overlap.  This is a measurement of how fast the SDR is
            changing.  This value is updated after every assignment to this SDR.
            This argument is NOT copied to any other SDRs, even by the copy
            constructor.  The given alpha is the weight given to each overlap.
            Giving a False value (including None and 0) will disable this
            feature entirely, it is OFF by default.  Attribute
            sdr.average_overlap contains the output, type=float, range=[0, 1].
        """
        # Private attribute self._callbacks is a list of functions of self, all
        # of which are called each time this SDR's value changes.
        self._callbacks = []
        if isinstance(specification, SDR):
            self.dimensions = specification.dimensions
            self.size       = specification.size
            self.assign(specification)
        else:
            self.dimensions = tuple(int(round(x)) for x in specification)
            self.size       = np.product(self.dimensions)
            self.zero()

        if activation_frequency_alpha is not None:
            self.activation_frequency_alpha = activation_frequency_alpha
            self.activation_frequency       = np.zeros(self.dimensions, dtype=np.float32)
            self._callbacks.append(type(self)._track_activation_frequency)

        if average_overlap_alpha is not None:
            self.average_overlap_alpha = average_overlap_alpha
            self.average_overlap       = 0.
            self._prev_value           = SDR(self)
            self._callbacks.append(type(self)._track_average_overlap)

    @property
    def dense(self):
        if self._dense is None:
            nz_values = self.nz_values
            if self._flat_index is not None:
                self._dense = np.zeros(self.size, dtype=np.uint8)
                self._dense[self._flat_index] = nz_values
                self._dense.shape = self.dimensions
            elif self._index is not None:
                self._dense = np.zeros(self.dimensions, dtype=np.uint8)
                self._dense[self.index] = nz_values
        return self._dense
    @dense.setter
    def dense(self, value):
        assert(isinstance(value, np.ndarray))
        value.shape      = self.dimensions
        assert(value.dtype.itemsize == 1)
        value.dtype      = np.uint8
        self._dense      = value
        self._index      = None
        self._flat_index = None
        self._nz_values  = None
        self._handle_callbacks()

    @property
    def index(self):
        if self._index is None:
            if self._flat_index is not None:
                self._index = np.unravel_index(self._flat_index, self.dimensions)
            elif self._dense is not None:
                self._index = np.nonzero(self._dense)
        return self._index
    @index.setter
    def index(self, value):
        value = tuple(value)
        assert(len(value) == len(self.dimensions))
        assert(all(idx.shape == value[0].shape for idx in value))
        assert(all(len(idx.shape) == 1 for idx in value))
        np_int_check = lambda idx: idx.dtype == np.int32 or idx.dtype == np.int64
        assert(all(np_int_check(idx) for idx in value))
        self._dense      = None
        self._index      = value
        self._flat_index = None
        self._nz_values  = None
        self._handle_callbacks()

    @property
    def flat_index(self):
        if self._flat_index is None:
            if self._index is not None:
                self._flat_index = np.ravel_multi_index(self._index, self.dimensions)
            elif self._dense is not None:
                self._flat_index = np.nonzero(self._dense.reshape(-1))[0]
        return self._flat_index
    @flat_index.setter
    def flat_index(self, value):
        assert(isinstance(value, np.ndarray))
        assert(len(value.shape) == 1)
        assert(value.dtype == np.int64 or value.dtype == np.int32)
        self._dense      = None
        self._index      = None
        self._flat_index = value
        self._nz_values  = None
        self._handle_callbacks()

    @property
    def nz_values(self):
        if self._nz_values is None:
            dense = self._dense
            if dense is not None:
                self._nz_values  = dense.reshape(-1)[self.flat_index]
            else:
                self._nz_values = np.ones(len(self), dtype=np.uint8)
        return self._nz_values
    @nz_values.setter
    def nz_values(self, value):
        assert(self._dense is None)
        assert(isinstance(value, np.ndarray))
        assert(value.shape == (len(self),))
        assert(value.dtype.itemsize == 1)
        self._nz_values       = value
        self._nz_values.dtype = np.uint8

    def assign(self, sdr):
        """
        Accepts an argument of unknown type and assigns it into this SDRs current
        value.  This accepts an SDR instance, dense boolean array, index tuple,
        flat index, or None.  If None is given then this takes no action and
        retains its current value.
        """
        if sdr is None:
            return
        if sdr is self:
            return
        self._dense      = None
        self._index      = None
        self._flat_index = None
        self._nz_values  = None

        if isinstance(sdr, SDR):
            assert(self.dimensions == sdr.dimensions)
            if sdr._dense is not None:
                self._dense = sdr._dense
            if sdr._index is not None:
                self._index = sdr._index
            if sdr._flat_index is not None:
                self._flat_index = sdr._flat_index
            if sdr._nz_values is not None:
                self._nz_values = sdr._nz_values
            self._handle_callbacks()

        elif isinstance(sdr, tuple):
            self.index = sdr

        elif isinstance(sdr, np.ndarray):
            if sdr.dtype == np.uint8 and self.size == np.product(sdr.shape):
                self.dense = sdr
            elif len(sdr.shape) == 1:
                self.flat_index = sdr

        if self._dense is None and self._index is None and self._flat_index is None:
            raise TypeError("Could not assign %s into an SDR."%type(sdr).__name__)

    def _handle_callbacks(self):
        for func in self._callbacks:
            func(self)

    def __len__(self):
        """Returns the number of active bits in current SDR."""
        return len(self.flat_index)

    def zero(self):
        """Sets all bits in the current sdr to zero."""
        self._dense      = None
        self._index      = None
        self._flat_index = np.empty(0, dtype=np.int)
        self._nz_values  = None

    def _track_activation_frequency(self):
        alpha = self.activation_frequency_alpha
        # Decay with time and incorperate this sample.
        self.activation_frequency             *= (1 - alpha)
        # Don't include NZ_Values, first because it's not correct and second
        # because it is set to None at the point where this is called.
        self.activation_frequency[self.index] += alpha

    def _track_average_overlap(self):
        alpha                = self.average_overlap_alpha
        overlap              = self.overlap(self._prev_value)
        self.average_overlap = (1 - alpha) * self.average_overlap + alpha * overlap
        self._prev_value     = SDR(self)

    def _dead_cell_filter(self):
        if self._index is not None:
            # Convert to flat index for this operation.
            self.flat_index
            self._index = None

        if self._flat_index is not None:
            self._flat_index = np.setdiff1d(self._flat_index, self._dead_cells)
            assert(self._nz_values is None)

        if self._dense is not None:
            self._dense.shape = -1
            self._dense[self._dead_cells] = 0
            self._dense.shape = self.dimensions

    def assign_flat_concatenate(self, sdrs):
        """Flats and joins its inputs, assigns the result to its current value."""
        sdrs = tuple(sdrs)
        assert(all(isinstance(s, SDR) for s in sdrs))
        assert(sum(s.size for s in sdrs) == self.size)
        self._dense      = None
        self._index      = None
        self._flat_index = np.empty(0, dtype=np.int)
        self._nz_values  = np.empty(0, dtype=np.uint8)
        offset = 0
        for sdr in sdrs:
            self._flat_index = np.concatenate([self._flat_index, sdr.flat_index + offset])
            self._nz_values  = np.concatenate([self._nz_values, sdr.nz_values])
            offset += sdr.size
        self._handle_callbacks()

    def slice_into(self, sdrs):
        """
        This divides this SDR's current value into peices and gives a peice to
        each of the given SDRs.  All SDRs must be 1 dimension.
        """
        sdrs = tuple(sdrs)
        assert(all(isinstance(s, SDR) for s in sdrs))
        assert(len(self.dimensions) == 1)
        assert(all(len(s.dimensions) == 1 for s in sdrs))
        assert(sum(s.size for s in sdrs) == self.size)

        offset = 0
        for slice_sdr in sdrs:
            slice_sdr.dense = self.dense[offset: offset + slice_sdr.size]
            offset += slice_sdr.size

    def overlap(self, other_sdr):
        """
        Documentation ...
        Explain that this is a measure of semantic similarity between two SDRs.

        Argument other_sdr is assigned into an SDR with the same dimensions as
                 this SDR, see SDR.assign for more information.

        Returns a number in the range [0, 1]
        """
        other = SDR(self.dimensions)
        other.assign(other_sdr)
        total_bits = np.sum(self.nz_values) + np.sum(other.nz_values)
        if total_bits == 0:
            return 0
        diff_bits  = np.array(self.dense, dtype=np.int)
        diff_bits -= other.dense
        diff_bits  = np.sum(np.abs(diff_bits))
        return (total_bits - diff_bits) / total_bits

    def false_positive_rate(self, active_sample_size, overlap_threshold):
        """
        Returns the theoretical false positive rate for a dendritic segment
        detecting the current value of this SDR.  This returns the probabilty
        random noise will activate the dendritic segment.

        Argument active_sample_size is the number of active bits which are
                 sampled onto the segment.
        Argument overlap_threshold is how many active bits are needed to
                 depolarize the segment.
        Argument self, this uses the current number of active bits, len(self) or
                 the mean activation frequency if it is available.

        Source: arXiv:1601.00720 [q-bio.NC], equation 4.
        """
        overlap_threshold = math.ceil(overlap_threshold)
        if hasattr(self, 'activation_frequency'):
            num_active_bits = int(round(np.mean(self.activation_frequency) * self.size))
        else:
            num_active_bits = len(self)
        # Can't sample more bits than are active.
        if active_sample_size > num_active_bits:
            return float('nan')

        num_inactive_bits = self.size - num_active_bits
        # Overlap set size is number of possible values for this SDR which
        # this segment could falsely detect.
        overlap_size = 0
        for overlap in range(overlap_threshold, active_sample_size+1):
            overlap_size += (_choose(num_active_bits, overlap)
                           * _choose(num_inactive_bits, active_sample_size - overlap))
        overlap_size -= 1 # For the intended pattern.
        # Divide by the number of different things which this segment could
        # possibly detect.
        num_possible_detections = _choose(self.size, active_sample_size)
        return Fraction(overlap_size, num_possible_detections)

    def false_negative_rate(self, missing_activations, active_sample_size, overlap_threshold):
        """
        Returns the theoretical false negative rate for a dendritic segment
        detecting the current value of this SDR.  This returns the probability
        that a segment which would normally detect the value will fail when some
        activations are supressed.

        Argument missing_activations is the fraction of neuron activations which
                 are missing from this SDR.
        Argument active_sample_size is how many active inputs are used to detect
                 the value.
        Argument overlap_threshold is how many active inputs are needed to
                 depolarize the segment, resulting in a prediction.
        Argument self, this uses the current number of active bits, len(self) or
                 the mean activation frequency if it is available.

        Source: arXiv:1601.00720 [q-bio.NC], equation 6.
        """
        overlap_threshold = math.ceil(overlap_threshold)
        if hasattr(self, 'activation_frequency'):
            num_active_bits = int(round(np.mean(self.activation_frequency) * self.size))
        else:
            num_active_bits = len(self)
        if active_sample_size < overlap_threshold:
            # The segment will never activate.
            return 1.
        # Can't sample more bits than are active.
        if active_sample_size > num_active_bits:
            return float('nan')
        assert(0 <= missing_activations <= 1) # missing_activations is the fraction of all activations.
        missing_activations = int(round(missing_activations * num_active_bits))
        # Count how many ways there are to corrupt this SDR such that it would
        # not be detected.
        false_negatives = 0
        # The overlap is between the corrupted bits and the segment's sample of
        # active bits.
        for overlap in range(min(missing_activations, active_sample_size)):
            if active_sample_size - overlap >= overlap_threshold:
                # There are too few corrupted bits in the active sample to
                # possibly cause a false negative.
                continue
            # The first part is the number of ways which corrupted bits could
            # fall in this segment.  The second part is the number of ways in
            # which corrupted bits could fall outside this segment.
            false_negatives += (_choose(active_sample_size, overlap)
                              * _choose(num_active_bits - active_sample_size, missing_activations - overlap))
        # Divide by the total number of ways to corrupt this SDR.
        num_corruptions = _choose(num_active_bits, missing_activations)
        return Fraction(false_negatives, num_corruptions)

    def entropy(self):
        """
        Calculates the entropy of this SDRs activations.

        Result is normalized to range [0, 1]
        A value of 1 indicates that all bits are equally and fully utilized.
        """
        if not hasattr(self, 'activation_frequency'):
            raise TypeError('Can not calculate entropy unless activation frequency is enabled for SDR.')
        p = self.activation_frequency
        e = _binary_entropy(p) / _binary_entropy(np.mean(p))
        return e

    def add_noise(self, percent):
        """
        Moves the given percent of active bits to a new location.
        TODO: This doesn't preserve the values, changed values become all ones.
        """
        num_bits    = len(self)
        flip_bits   = int(round(len(self) * percent))
        self.flat_index     # Convert current value to flat index format.
        self._index = None
        self._dense = None
        # Turn off bits.
        off_bits_idx     = np.random.choice(num_bits, size=flip_bits, replace=False)
        off_bits         = self._flat_index[off_bits_idx]
        self._flat_index = np.delete(self._flat_index, off_bits_idx)
        # Turn on bits
        on_bits = set(self._flat_index)
        for x in range(flip_bits):
            while True:
                bit = np.random.choice(self.size)
                if bit not in on_bits:
                    break
            on_bits.add(bit)
        self._flat_index = np.array(list(on_bits), dtype=np.int)

    def kill_cells(self, percent):
        """
        Short Description ...
        Argument percent ...
        """
        num_cells        = int(round(self.size * percent))
        self._dead_cells = np.random.choice(self.size, num_cells, replace=False)
        self._callbacks.append(type(self)._dead_cell_filter)

    def statistics(self):
        """Returns a string describing this SDR."""
        stats = 'SDR%s\n'%str(self.dimensions)

        if hasattr(self, 'average_overlap'):
            stats += '\tAverage Overlap %g\n'%self.average_overlap

        if hasattr(self, 'activation_frequency'):
            af = self.activation_frequency
            e  = np.nan_to_num(self.entropy())
            stats += '\tEntropy: %d%%\n'%round(e*100)
            stats += '\tActivation Frequency min/mean/std/max  %-.04g%% / %-.04g%% / %-.04g%% / %-.04g%%\n'%(
                np.min(af)*100,
                np.mean(af)*100,
                np.std(af)*100,
                np.max(af)*100,)
        else:
            stats += '\tCurrent Sparsity %.04g%%\n'%(100 * len(self) / self.size)

        return stats

SDR = SparseDistributedRepresentation


# TODO: Instead of calling assign-flat-concat, make and use this class
#   First write the docstrings for it...
class SDR_Concatenation(SDR):
    """

    SDR concatenations are read only.  To change its value assign to any of the
    component SDRs.  This SDRs value is computed and cached when it is accessed.
    """
    def __init__(self, concat_sdrs, axis=None):
        """
        """
        self.input_sdrs = tuple(concat_sdrs)
        assert(all(isinstance(x, SDR) for x in self.input_sdrs))
        concat_dim = sum(sdr.dimensions[axis] for sdr in self.input_sdrs)
        # assert(all(dims are ok excpt for axis which can be anything...))
        self.size = sum(x.size for x in self._concat_sdrs)
        self.dimensions = 1/0
        self.valid = False
        for sdr in self._concat_sdrs:
            sdr._callbacks.append(self._callback)

    def _callback(self):
        self.valid = False

    @property
    def dense(self):
        if self._dense is None:
            nz_values = self.nz_values
            if self._flat_index is not None:
                self._dense = np.zeros(self.size, dtype=np.uint8)
                self._dense[self._flat_index] = nz_values
                self._dense.shape = self.dimensions
            elif self._index is not None:
                self._dense = np.zeros(self.dimensions, dtype=np.uint8)
                self._dense[self.index] = nz_values
        return self._dense

    @property
    def index(self):
        if self._index is None:
            if self._flat_index is not None:
                self._index = np.unravel_index(self._flat_index, self.dimensions)
            elif self._dense is not None:
                self._index = np.nonzero(self._dense)
        return self._index

    @property
    def flat_index(self):
        if self._flat_index is None:
            if self._index is not None:
                self._flat_index = np.ravel_multi_index(self._index, self.dimensions)
            elif self._dense is not None:
                self._flat_index = np.nonzero(self._dense.reshape(-1))[0]
        return self._flat_index

    @property
    def nz_values(self):
        if self._nz_values is None:
            dense = self._dense
            if dense is not None:
                dense.shape = -1
                self._nz_values  = dense[self.flat_index]
                dense.shape = self.dimensions
            else:
                self._nz_values = np.ones(len(self), dtype=np.uint8)
        return self._nz_values


class SDR_Subsample(SDR):
    """
    Makes an SDR smaller by randomly removing bits.

    This class can also consolidate topological dimensions by lumping blocks of
    the input SDR into a single topological area of the output SDR.

    SDR subsamples are read only.  Assignment to the input SDR clears this SDR's
    value and this SDR's value is computed and cached when it is accessed.
    """
    def __init__(self, input_sdr, dimensions):
        """
        Argument input_sdr is an instance of SDR ...

        Argument dimensions is a tuple of N integers.  This tuple is divided
            into the first N-1 numbers which represent the topological
            dimensions and the final number which represents all of the extra
            dimensions.  The input SDR is assumed to have the same number of
            topological dimensions as this subsample.  This class resizes the
            topological dimensions from the input SDR to the output SDR's
            dimensions and lumps and inputs which are condensed into an area
            into the same extra dimension.  Then the extra dimension is
            subsampled so that there are as many inputs in each topological area
            as the argument dimensions[-1] specifies.

        Example:  SDR_Subsample(SDR((200, 200, 2)), (8, 8, 400))
        In this example there are 2 topological dimensions.

        First this reshapes the topological dimensions from (200, 200) to (8, 8)
        by cutting the input SDR into rectangles of size (200/8, 200/8) which
        equals (25, 25).  Each (25, 25) rectangle of the input space is
        extracted, flattened and stored as a single point in the output spaces
        (8, 8) grid, yielding a shape of (8, 8, 25*25*2) which equals
        (8, 8, 1250).  

        Second this takes a random sample of the extra dimension, and discards
        the remainder of it.  This brings the shape from (8, 8, 1250) to
        (8, 8, 400).
        """
        SDR.__init__(self, dimensions)
        assert(isinstance(input_sdr, SDR))
        self.input_sdr = input_sdr
        self.input_sdr._callbacks.append(self._clear)
        # This SDR can not be assigned to so using _callbacks doesn't make
        # sense. If anything, use the input_sdr's _callbacks list.
        self._callbacks = None

        # Determine how to chop up the input space into evenly-sized rectangles.
        area_slices = []
        nd_topological = len(self.dimensions) - 1
        for dim in range(nd_topological):
            inp_dim_sz = self.input_sdr.dimensions[dim]
            out_dim_sz = self.dimensions[dim]
            bounds     = np.linspace(0, inp_dim_sz, out_dim_sz + 1)
            bounds     = np.array(np.rint(bounds), dtype=np.int)
            slices     = [slice(st, end) for st, end in zip(bounds, bounds[1:])]
            area_slices.append(slices)

        # Internaly this uses an array of flat indices specifying which input
        # goes to each output location.
        self.sources   = np.empty(self.size, dtype=np.int)
        input_indices  = np.arange(self.input_sdr.size)
        input_indices  = input_indices.reshape(self.input_sdr.dimensions)
        # Flatten the topological dimensions for easier access.
        subsample_size = self.dimensions[-1]
        self.sources   = self.sources.reshape(-1, subsample_size)

        # Iterate through every topological output area and sample some inputs.
        for out_topo_idx, slice_tuple in enumerate(itertools.product(*area_slices)):
            potential_sources = input_indices[slice_tuple]
            potential_sources = potential_sources.flatten()
            subsample_sources = np.random.choice(
                potential_sources,
                subsample_size,
                replace=False)
            self.sources[out_topo_idx] = subsample_sources
        self.sources = self.sources.reshape(self.dimensions)

    def _clear(self):
        self._dense      = None
        self._index      = None
        self._flat_index = None
        self._nz_values  = None

    @property
    def dense(self):
        if self._dense is None:
            self._dense = self.input_sdr.dense[self.sources]
        return self._dense

    @property
    def index(self):
        if self._index is None:
            self._index = np.nonzero(self.dense)
        return self._index

    @property
    def flat_index(self):
        if self._flat_index is None:
            self._flat_index = np.nonzero(self.dense.reshape(-1))[0]
        return self._flat_index

    @property
    def nz_values(self):
        if self._nz_values is None:
            self._nz_values = self.dense.reshape(-1)[self.flat_index]
        return self._nz_values
