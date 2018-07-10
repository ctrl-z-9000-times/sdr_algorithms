#!/usr/bin/python3
"""
Functional Test: ASCII-TM
Written by David McDougall, 2018

This tests that the TemporalMemory can recognize sequences of inputs.  The
system consists of an encoder, spatial pooler, temporal memory, and SDR
classifier. The system is shown each word in the dataset in isolation, and is
reset before seeing each word.  The classifier is trained and tested on the
final character of the word.
"""

import argparse
import random
import numpy as np

import sys
sys.path.append('.')
from sdr import SDR
from encoders import EnumEncoder
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory
from nupic.algorithms.sdr_classifier import SDRClassifier
from synapses import debug as synapses_debug

dictionary_file = '/usr/share/dict/american-english'

def read_dictionary():
    with open(dictionary_file) as f:
        dictionary = f.read().split()
    # Reject apostrophies from the dictionary.  Words with apostrophies are
    # mostly duplicates and this makes the printouts look nicer.
    dictionary = [word for word in dictionary if "'" not in word]
    return dictionary

state_names = [
    'ALABAMA',
    'ALASKA',
    'ARIZONA',
    'ARKANSAS',
    'CALIFORNIA',
    'COLORADO',
    'CONNECTICUT',
    'DELAWARE',
    'FLORIDA',
    'GEORGIA',
    'HAWAII',
    'IDAHO',
    'ILLINOIS',
    'INDIANA',
    'IOWA',
    'KANSAS',
    'KENTUCKY',
    'LOUISIANA',
    'MAINE',
    'MARYLAND',
    'MASSACHUSETTS',
    'MICHIGAN',
    'MINNESOTA',
    'MISSISSIPPI',
    'MISSOURI',
    'MONTANA',
    'NEBRASKA',
    'NEVADA',
    'NEW HAMPSHIRE',
    'NEW JERSEY',
    'NEW MEXICO',
    'NEW YORK',
    'NORTH CAROLINA',
    'NORTH DAKOTA',
    'OHIO',
    'OKLAHOMA',
    'OREGON',
    'PENNSYLVANIA',
    'RHODE ISLAND',
    'SOUTH CAROLINA',
    'SOUTH DAKOTA',
    'TENNESSEE',
    'TEXAS',
    'UTAH',
    'VERMONT',
    'VIRGINIA',
    'WASHINGTON',
    'WEST VIRGINIA',
    'WISCONSIN',
    'WYOMING',
]

default_parameters = {
          'enc': {'size': 5102, 'sparsity': 0.017973130662032412},
          'sp': {'mini_columns': 1225,
                 'permanence_dec': 0.0028900881528214164,
                 'permanence_inc': 0.021027522837449332,
                 'permanence_thresh': 0.024721604498563178,
                 'potential_pool': 3298,
                 'sparsity': 0.03775592629273584},
          'tm': {'add_synapses': 31,
                 'cells_per_column': 19,
                 'init_dist': (0.7167876106409492, 0.0),
                 'learning_threshold': 9,
                 'mispredict_dec': 0.0023680534135813087,
                 'permanence_dec': 0.005098356000496391,
                 'permanence_inc': 0.04597764517227905,
                 'permanence_thresh': 0.23155288819735864,
                 'predicted_boost': 3,
                 'predictive_threshold': 24,
                 'segments_per_cell': 26,
                 'synapses_per_segment': 37}}

def main(parameters=default_parameters, argv=None, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=int, default=20,
                        help='Number of times to run through the training data.')
    parser.add_argument('--dataset', choices=('states', 'dictionary'), default='states')
    args = parser.parse_args(args = argv)

    # Load data.
    if args.dataset == 'states':
        dataset = state_names
        print("Dataset is %d state names"%len(dataset))
    elif args.dataset == 'dictionary':
        dataset = read_dictionary()
        dataset = random.sample(dataset, 200)
        if verbose:
            print("Dataset is dictionary words, sample size %d"%len(dataset))

    dataset   = [word.upper() for word in dataset]
    dataset   = sorted(dataset)
    word_ids  = {word: idx for idx, word in enumerate(sorted(dataset))}
    confusion = np.zeros((len(dataset), len(dataset)))
    if verbose:
        print("Dataset: " + ", ".join('%d) %s'%idx_word for idx_word in enumerate(dataset)))

    # Construct TM.
    enc = EnumEncoder(**parameters['enc'])
    enc.output_sdr = SDR(enc.output_sdr, average_overlap_alpha = 1./1000)
    sp = SpatialPooler(
        input_sdr         = enc.output_sdr,
        boosting_alpha    = 1./1000,
        **parameters['sp'])
    tm = TemporalMemory(
        column_sdr        = sp.columns,
        anomaly_alpha     = 1/1000,
        **parameters['tm'])
    sdrc = SDRClassifier(steps=[0])

    def reset():
        enc.output_sdr.zero()
        sp.reset()
        tm.reset()

    # Train.
    if verbose:
        print("Training for %d cycles"%(args.time * sum(len(w) for w in dataset)))
    t = 0
    for i in range(args.time):
        random.shuffle(dataset)
        for word in dataset:
            reset()
            for idx, char in enumerate(word):
                enc.encode(char)
                sp.compute()
                tm.compute()
            lbl = word_ids[word]
            sdrc.compute(t, tm.active.flat_index,
                classification={"bucketIdx": lbl, "actValue": lbl},
                learn=True, infer=False)
            t += 1

    if verbose:
        print("Encoder", enc.output_sdr.statistics())
        print(sp.statistics())
        print(tm.statistics())

    # Test.
    score = 0.
    score_samples = 0
    for word in dataset:
        reset()
        for idx, char in enumerate(word):
            enc.encode(char)
            sp.compute(learn = False)
            tm.compute(learn = False)

        inference = sdrc.infer(tm.active.flat_index, None)
        lbl = word_ids[word]
        if lbl == np.argmax(inference[0]):
            score += 1
        score_samples += 1
        confusion[lbl] += inference[0]
    print("Score:", 100. * score / score_samples, '%')

    if synapses_debug:
        tm.synapses.check_data_integrity()
        print("Synapse data structure integrity is OK.")

    if verbose:
        import matplotlib.pyplot as plt
        plt.figure('Confusion Matrix')
        plt.imshow(confusion, interpolation='nearest')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

    return score / score_samples

if __name__ == '__main__':
    main()
