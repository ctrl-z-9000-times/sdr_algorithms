#!/usr/bin/python3
"""
Functional Test: STABILITY
Written by David McDougall, 2018

This verifies basic properties of the Stable Spatial Pooler.  This generates a
new artificial dataset every time.  The dataset consists of randomly generated
SDRs which are fed into the system as a timeseries.  The dataset represents
objects. Objects are composed of non-mutually-exclusive sets of inputs.  Each
computation cycle the system sees part of an object, and the system sees
multiple parts of an object on consequtive cycles.  The test is to form a single
stable representation of each object, despite each object being composed of
disimilar components.  The second test is to recognise each object as it is
seen.
"""

import numpy as np
import itertools
import random
import math

import sys
sys.path.append('.')
from sdr import SDR
from encoders import EnumEncoder
from spatial_pooler import StableSpatialPooler
from nupic.algorithms.sdr_classifier import SDRClassifier
from synapses import debug as synapses_debug

def object_dataset(num_objects, object_sizes):
    alphabet = [chr(ord('A') + x) for x in range(26)]
    inputs   = [''.join(chrs) for chrs in itertools.product(alphabet, repeat=3)]
    # objects  = [random.sample(inputs, random.choice(object_sizes)) for x in range(num_objects)]
    objects  = []
    for object_id in range(num_objects):
        objects.append([])
        for aspect in range(random.choice(object_sizes)):
            objects[-1].append(random.choice(inputs))
    return inputs, objects

def measure_inter_intra_overlap(catagories, verbose = True):
    """
    Argument catagories is a dictionary of lists of SDRs, where the keys are
        the data labels, and the values are all of the sampled activation
        pattern from the catagory.
    """
    if isinstance(catagories, dict):
        catagories = list(catagories.values())
    # Shuffle all of the samples so that they can be safely discarded when
    # enough have been used.
    for sdr_vec in catagories:
        random.shuffle(sdr_vec)

    n_samples = 1e6

    # Measure average overlap within categories.
    stability         = 0
    stability_samples = 0
    for obj_samples in catagories:
        catagory_samples = 0
        for sdr1, sdr2 in itertools.combinations(obj_samples, 2):
            stability         += sdr1.overlap(sdr2)
            stability_samples += 1
            catagory_samples  += 1
            if catagory_samples > n_samples / len(catagories):
                break
    if stability_samples == 0:
        stability_samples = 1
        print("Warning: stability_samples == 0")
    stability = stability / stability_samples
    if verbose:
        print('Intra Category Overlap %g%% (%d samples)'%(100*stability, stability_samples))

    # Measure average overlap between categories.
    distinctiveness         = 0
    distinctiveness_samples = 0
    n_combos  = len(catagories) * (len(catagories) - 1) / 2
    subsample = int( (n_samples / n_combos) ** .5 )
    for obj1_samples, obj2_samples in itertools.combinations(catagories, 2):
        for sdr1 in obj1_samples[ : subsample]:
            for sdr2 in obj2_samples[ : subsample]:
                distinctiveness         += sdr1.overlap(sdr2)
                distinctiveness_samples += 1
    if distinctiveness_samples == 0:
        distinctiveness_samples = 1
        print("Warning: distinctiveness_samples == 0")
    distinctiveness = distinctiveness / distinctiveness_samples

    try:
        stability_metric = stability / distinctiveness
    except ZeroDivisionError:
        stability_metric = float('nan')
    if verbose:
        print('Inter Category Overlap %g%% (%d samples)'%(100*distinctiveness, distinctiveness_samples))
        print('Stability Metric',       stability_metric)
    return stability, distinctiveness, stability_metric

default_parameters = {
          'active_thresh': 10,
          'boosting_alpha': 0.005694742035947877,
          'mini_columns': 2112,
          'permanence_dec': 0.0034674259121478907,
          'permanence_inc': 0.06557477182718344,
          'permanence_thresh': 0.11140401898422288,
          'potential_pool': 2010,
          'segments': 6,
          'sparsity': 0.026277223129229404,
          'stability_rate': 0.06921208621873447}

def main(parameters=default_parameters, argv=None, verbose=True):
    # Setup
    num_objects        = 100
    object_sizes       = range(20, 40+1)
    train_iterations   = 100
    test_iterations    = 5
    steps_per_object   = range(3, 17+1)
    inputs, objects = object_dataset(num_objects, object_sizes)

    enc = EnumEncoder(2400, 0.02)
    enc.output_sdr = SDR(enc.output_sdr,
        activation_frequency_alpha = parameters['boosting_alpha'],
        average_overlap_alpha      = parameters['boosting_alpha'],)

    sp = StableSpatialPooler(
        input_sdr         = enc.output_sdr,
        macro_columns     = (1,),
        **parameters)
    sdrc = SDRClassifier(steps=[0])

    def measure_catagories():
        # Compute every sensation for every object.
        objects_columns = []
        for obj in objects:
            objects_columns.append([])
            for sensation in obj:
                sp.reset()
                enc.encode(sensation)
                sp.compute(learn=False)
                objects_columns[-1].append(SDR(sp.columns))
        sp.reset()
        return objects_columns

    if verbose:
        print("Num-Inputs  ", len(set(itertools.chain.from_iterable(objects))))
        print('Num-Objects ', num_objects)
        print("Object-Sizes", object_sizes)
        print("Steps/Object", steps_per_object)
        print(sp.statistics())
        objects_columns = measure_catagories()
        measure_inter_intra_overlap(objects_columns, verbose)
        print("")

        # TRAIN
        train_time = train_iterations * num_objects * np.mean(steps_per_object)
        print('TRAINING for ~%d Cycles (%d dataset iterations) ...'%(train_time, train_iterations))
        print("")

    sp.reset()
    t = 0
    for iteration in range(train_iterations):
        object_order = list(range(num_objects))
        random.shuffle(object_order)
        for object_id in object_order:
            for step in range(random.choice(steps_per_object)):
                sensation = random.choice(objects[object_id])
                enc.encode(sensation)
                sp.compute()
                try:
                    sdrc.compute(t, sp.columns.flat_index,
                        classification = {"bucketIdx": object_id, "actValue": object_id,},
                        learn=True, infer=False)
                except ValueError:
                    print("Warning: len(active) = %d."%(len(sp.columns)))
                t += 1

    if verbose:
        print("TESTING ...")
        print("")
        print('Encoder Output', enc.output_sdr.statistics())
        print(sp.statistics())

    objects_columns = measure_catagories()
    _, __, stability_metric = measure_inter_intra_overlap(objects_columns, verbose)

    # Measure classification accuracy.  This test consists of looking at every
    # object a few times and then classifying it.  The AI is evaluated on every
    # cycle.
    score = 0
    max_score = 0
    sp.reset()
    if verbose:
        print("")
        print("Test length: %d dataset iterations."%(test_iterations))
    test_data = list(range(num_objects))
    for iteration in range(test_iterations):
        random.shuffle(test_data)
        for object_id in test_data:
            for step in range(random.choice(steps_per_object)):
                sensation = random.choice(objects[object_id])
                enc.encode(sensation)
                sp.compute(learn=True)
                inference = sdrc.infer(sp.columns.flat_index, None)[0]
                inference = np.argmax(inference)
                if inference == object_id:
                    score += 1
                max_score += 1
    if verbose:
        print('Classification Accuracy: %g %%'%(100 * score / max_score))

    if synapses_debug:
        sp.synapses.check_data_integrity()
        print("Synapse data structure integrity is OK.")

    return stability_metric + 10 * (score / max_score)

if __name__ == '__main__':
    main()
