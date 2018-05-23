# Written by David McDougall, 2017

import numpy as np
import random
from sdr import SDR


# TODO: This should use or at least print the radius, ie the distance at which
# two numbers will have 50% overlap.  Radius is a replacement for resolution.
class RandomDistributedScalarEncoder:
    """https://arxiv.org/pdf/1602.05925.pdf"""
    def __init__(self, resolution, size, sparsity):
        self.resolution = resolution
        self.output_sdr = SDR((size,))
        self.sparsity   = sparsity

    def encode(self, value):
        # This must be integer division! Everything under the resolution must be removed.
        index   = value // self.resolution
        size    = self.output_sdr.size
        code    = np.zeros(size, dtype=np.uint8)
        on_bits = int(round(size * self.sparsity))
        for offset in range(on_bits):
            # Cast to string before hash, python3 will not hash an integer, uses
            # value instead.
            h = hash(str(index + offset))
            bucket = h % size
            # If this bucket is already full, walk around until it finds one
            # that isn't taken.
            while code[bucket]:
                bucket = (bucket + 1) % size
            code[bucket] = 1
        self.output_sdr.dense = code
        return self.output_sdr


class EnumEncoder:
    """
    Encodes arbirary enumerated values.
    There is no semantic similarity between encoded values.

    This encoder associates values with SDRs.  It works by hashing the value,
    seeding a pseudo-random number generator with the hash, and activating a
    random sample of the bits in the output SDR.
    """
    def __init__(self, size, sparsity):
        self.output_sdr   = SDR((size,))
        self.sparsity     = sparsity

    def encode(self, value):
        """
        Argument value must be hashable.
        Returns SDR
        """
        num_active = int(round(self.output_sdr.size * self.sparsity))
        enum_rng   = random.Random(hash(value))
        active     = enum_rng.sample(range(self.output_sdr.size), num_active)
        self.output_sdr.flat_index = np.array(active)
        return self.output_sdr


class ChannelEncoder:
    """
    This assigns a random range to each bit of the output SDR.  Each bit becomes
    active if its corresponding input falls in its range.  By using random
    ranges, each bit represents a different thing even if it mostly overlaps
    with other comparable bits.  This way redundant bits add meaning.
    """
    def __init__(self, input_shape, num_samples, sparsity,
        dtype       = np.float64,
        drange      = range(0,1),
        wrap        = False):
        """
        Argument input_shape is tuple of dimensions for each input frame.

        Argument num_samples is number of bits in the output SDR which will
                 represent each input number, this is the added data depth.

        Argument sparsity is fraction of output which on average will be active.
                 This is also the fraction of the input spaces which (on 
                 average) each bin covers.

        Argument dtype is numpy data type of channel.

        Argument drange is a range object or a pair of values representing the 
                 range of possible channel values.

        Argument wrap ... default is False.
                 This supports modular input spaces and ranges which wrap
                 around. It does this by rotating the inputs by a constant
                 random amount which hides where the discontinuity in ranges is.
                 No ranges actually wrap around the input space.
        """
        self.input_shape  = tuple(input_shape)
        self.num_samples  = int(round(num_samples))
        self.sparsity     = sparsity
        self.output_shape = self.input_shape + (self.num_samples,)
        self.dtype        = dtype
        self.drange       = drange
        self.len_drange   = max(drange) - min(drange)
        self.wrap         = bool(wrap)
        if self.wrap:
            self.offsets  = np.random.uniform(0, self.len_drange, self.input_shape)
            self.offsets  = np.array(self.offsets, dtype=self.dtype)
        # Each bit responds to a range of input values, length of range is 2*Radius.
        radius            = self.len_drange * self.sparsity / 2
        if self.wrap:
            # If wrapping is enabled then don't generate ranges which will be
            # truncated near the edges.
            centers = np.random.uniform(min(self.drange) + radius,
                                        max(self.drange) - radius,
                                        size=self.output_shape)
        else:
            # Ranges within a radius of the edges are OK.  They will not respond
            # to a full range of input values but are needed to represent the
            # bits at the edges of the data range.
            centers = np.random.uniform(min(self.drange),
                                        max(self.drange),
                                        size=self.output_shape)
        # Make the lower and upper bounds of the ranges.
        self.low  = np.array(centers - radius, dtype=self.dtype)
        self.high = np.array(centers + radius, dtype=self.dtype)

    def encode(self, img):
        """Returns a dense boolean np.ndarray."""
        assert(img.shape == self.input_shape)
        assert(img.dtype == self.dtype)
        if self.wrap:
            img += self.offsets
            # Technically this should subtract min(drange) before doing modulus
            # but the results should also be indistinguishable B/C of the random
            # offsets.  Min(drange) effectively becomes part of the offset.
            img %= self.len_drange
            img += min(self.drange)
        img = img.reshape(img.shape + (1,))
        return np.logical_and(self.low <= img, img <= self.high)

    def __str__(self):
        lines = ["Channel Encoder,  num-samples %d"%int(round(self.args.num_samples))]
        lines.append("\tSparsity %.03g, dtype %s, drange %s %s"%(
                self.sparsity,
                self.dtype.__name__,
                self.drange,
                'Wrapped' if self.wrap else ''))
        return '\n'.join(lines)


class ChannelThresholder:
    """
    Creates a channel encoder with an additional activation threshold.  A bit
    becomes active if and only if the underlying channel encoder activates it
    and its magnitude is not less than its threshold. Activation thresholds are
    normally distributed.
    """
    def __init__(self, num_samples,
        sparsity,
        mean,
        stddev, input_shape, dtype, drange, wrap):
        """
        Argument num_samples ... see ChannelEncoder
        Argument sparsity ... see ChannelEncoder
        Argument mean is the average of activation thresholds.
        Argument stddev is the standard deviation of activation thresholds.
        Argument input_shape is tuple of dimensions of each input frame.
        Arguments dtype, drange, and wrap are passed through to the underlying
                  channel encoder.
        """
        1/0 # Unimplemented.
        assert(isinstance(parameters, ChannelThresholderParameters))
        self.args = args  = parameters
        self.channel      = ChannelEncoder(input_shape, args.num_samples, args.sparsity,
                            dtype=dtype, drange=drange, wrap=wrap)
        self.output_shape = self.channel.output_shape
        self.thresholds   = np.random.normal(args.mean, args.stddev, self.output_shape)
        self.thresholds   = np.array(self.thresholds, dtype)

    def encode(self, img_data, magnitude):
        """
        Send raw data and magnitudes, this runs the channel encoder as well as
        the thresholder.
        """
        sdr = self.channel.encode(img_data)
        assert(magnitude.shape == self.channel.input_shape)
        # Reshape to broadcast magnitudes across the data dimension to all
        # samples and their thresholds.
        magnitude = magnitude.reshape(magnitude.shape + (1,))
        sdr[magnitude < self.thresholds] = False
        return sdr
