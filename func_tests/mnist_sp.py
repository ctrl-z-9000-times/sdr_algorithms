#!/usr/bin/python3
"""
Written by David McDougall, 2018

Functional Test: MNIST-SP

This tests that the spatial pooler can recognise basic patterns in its input.
The system consists of a simple black & white image encoder, a spatial pool, and
an SDR classifier.  The task is to recognise images of hand written numbers 0-9.
This should score at least 95%.
"""

import argparse
import random
import gzip
import scipy.ndimage
import numpy as np

import sys
sys.path.append('.')
from sdr import SDR
from spatial_pooler import SpatialPooler
from nupic.algorithms.sdr_classifier import SDRClassifier
from synapses import debug as synapses_debug

def load_mnist():
    """See: http://yann.lecun.com/exdb/mnist/ for MNIST download and binary file format spec."""
    def int32(b):
        i = 0
        for char in b:
            i *= 256
            # i += ord(char)    # python2
            i += char
        return i

    def load_labels(file_name):
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2049)  # Magic number
            labels = []
            for char in raw[8:]:
                # labels.append(ord(char))      # python2
                labels.append(char)
        return labels

    def load_images(file_name):
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2051)    # Magic number
            num_imgs   = int32(raw[4:8])
            rows       = int32(raw[8:12])
            cols       = int32(raw[12:16])
            assert(rows == 28)
            assert(cols == 28)
            img_size   = rows*cols
            data_start = 4*4
            imgs = []
            for img_index in range(num_imgs):
                vec = raw[data_start + img_index*img_size : data_start + (img_index+1)*img_size]
                # vec = [ord(c) for c in vec]   # python2
                vec = list(vec)
                vec = np.array(vec, dtype=np.uint8)
                buf = np.reshape(vec, (rows, cols, 1))
                imgs.append(buf)
            assert(len(raw) == data_start + img_size * num_imgs)   # All data should be used.
        return imgs

    train_labels = load_labels('func_tests/MNIST_data/train-labels-idx1-ubyte.gz')
    train_images = load_images('func_tests/MNIST_data/train-images-idx3-ubyte.gz')
    test_labels  = load_labels('func_tests/MNIST_data/t10k-labels-idx1-ubyte.gz')
    test_images  = load_images('func_tests/MNIST_data/t10k-images-idx3-ubyte.gz')

    return train_labels, train_images, test_labels, test_images

def synthesize(seed, diag=False):
    """
    Modify an image with random shifts, scales, and rotations.
    Use this function to expand the training dataset and make it more robust to these transforms.

    Note: translation is worse for training MNIST b/c the test set is centered.
    Translation just makes the problem harder.

    TODO: Stretching/scaling/skewing images
    """
    # Apply a random rotation
    theta_max = 5      # degrees
    theta = random.uniform(-theta_max, theta_max)
    synth = scipy.ndimage.interpolation.rotate(seed, theta, order=0, reshape=False)

    def bounding_box(img):
        # Find the bounding box of the character
        r_occupied = np.sum(img, axis=1)
        for r_min in range(len(r_occupied)):
            if r_occupied[r_min]:
                break
        for r_max in range(len(r_occupied)-1, -1, -1):
            if r_occupied[r_max]:
                break

        c_occupied = np.sum(img, axis=0)
        for c_min in range(len(c_occupied)):
            if c_occupied[c_min]:
                break
        for c_max in range(len(c_occupied)-1, -1, -1):
            if c_occupied[c_max]:
                break
        return r_min, r_max, c_min, c_max

    # Stretch the image in a random direction
    pass

    if False:
        # Apply a random shift
        r_min, r_max, c_min, c_max = bounding_box(synth)
        r_shift = random.randint(-r_min, len(r_occupied) -1 -r_max)
        c_shift = random.randint(-c_min, len(c_occupied) -1 -c_max)
        synth = scipy.ndimage.interpolation.shift(synth, [r_shift, c_shift, 0])

    if diag:
        from matplotlib import pyplot as plt
        plt.figure(1)
        sz = 3
        example_synths = [synthesize(seed, diag=False) for _ in range(sz**2 - 2)]
        example_synths.append(synth)
        plt.subplot(sz, sz, 1)
        plt.imshow(np.dstack([seed/255]*3), interpolation='nearest')
        plt.title("Seed")
        for i, s in enumerate(example_synths):
            plt.subplot(sz, sz, i+2)
            plt.imshow(np.dstack([s/255]*3), interpolation='nearest')
            plt.title("Synthetic")
        plt.show()

    return synth

class BWImageEncoder:
    """Simple grey scale image encoder for MNIST."""
    def __init__(self, input_space):
        self.output = SDR(tuple(input_space) + (2,))

    def encode(self, image):
        mean = np.mean(image)
        on_bits  = image >= mean
        off_bits = np.logical_not(on_bits)
        self.output.dense = np.dstack([on_bits, off_bits])
        return self.output

default_parameters = {
 'active_thresh': 28,
 'boosting_alpha': 0.0007133352820011685,
 'macro_columns': (15, 10),
 'mini_columns': 250,
 'permanence_dec': 0.009282175512186927,
 'permanence_inc': 0.032294958166728734,
 'permanence_thresh': 0.4222038828631455,
 'potential_pool': 106,
 'radii': (2.7582610908140706, 2.9070918151296423),
 'sparsity': 0.014091711495013154}

def main(parameters=default_parameters, argv=None, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=float, default=1,
                        help='Number of times to run through the training data.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(args = argv)

    # Load data.
    train_labels, train_images, test_labels, test_images = load_mnist()

    if False:
        # Experiment to verify that input dimensions are handled correctly If
        # you enable this, don't forget to rescale the radii as well as the
        # input.
        from scipy.ndimage import zoom
        new_sz = (1, 4, 1)
        train_images = [zoom(im, new_sz, order=0) for im in train_images]
        test_images  = [zoom(im, new_sz, order=0) for im in test_images]

    training_data = list(zip(train_images, train_labels))
    test_data     = list(zip(test_images, test_labels))
    random.shuffle(training_data)
    random.shuffle(test_data)
    if args.debug and args.time < 1:
        test_data = test_data[: int(len(test_data) * args.time)]
    # Setup spatial pooler machine.
    enc = BWImageEncoder(train_images[0].shape[:2])
    sp = SpatialPooler(
        input_sdr         = enc.output,
        segments          = 1,
        **parameters)
    sdrc = SDRClassifier(steps=[0])

    if verbose:
        print(sp.statistics())

    # Training Loop
    train_cycles = len(train_images) * args.time
    if verbose:
        print("Training for %d cycles"%train_cycles)
    for i in range(int(round(train_cycles))):
        sp.reset()
        img, lbl      = random.choice(training_data)
        img           = synthesize(img, diag=False)
        enc.encode(np.squeeze(img))
        sp.compute()
        sdrc.compute(i, sp.columns.flat_index,
            classification={"bucketIdx": lbl, "actValue": lbl},
            learn=True, infer=False)

    if verbose:
        print("Done training.")
        print("")
        print("Removing zero permanence synapses.")
        sp.synapses.remove_zero_permanence_synapses()
        print(sp.statistics())

    # Testing Loop
    if verbose:
        print("Testing for %d cycles."%len(test_data))
    score = 0
    for img, lbl in test_data:
        enc.encode(np.squeeze(img))
        sp.compute(learn=False)
        try:
            inference = sdrc.infer(sp.columns.flat_index, None)[0]
        except IndexError:
            inference = np.zeros(10)
        if lbl == np.argmax(inference):
            score += 1

    print('Score:', 100 * score / len(test_data), '%')

    if synapses_debug:
        sp.synapses.check_data_integrity()
        print("Synapse data structure integrity is OK.")

    return score / len(test_data)

if __name__ == '__main__':
    main()
