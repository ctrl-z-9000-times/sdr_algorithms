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
    # Convert to all capital letters.
    dictionary = [word.upper() for word in dictionary]
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
    'enc': {'size': 3896, 'sparsity': 0.011016943726056617},
      'sp': {'mini_columns': 2000,    # MODIFIED
             'permanence_dec': 0.0030453258771592872,
             'permanence_inc': 0.01735636057973911,
             'permanence_thresh': 0.039191950857707235,
             'potential_pool': 2613,
             'sparsity': 0.06306098753882936,
            'boosting_alpha' : 1./100,},    # MODIFIED
      'tm': {'add_synapses': 23,
             'cells_per_column': 32,    # MODIFIED
             'init_dist': (0.5416786696673432, 0.010625659253282042),
             'learning_threshold': 9,
             'mispredict_dec': 0.002446443463633688,
             'permanence_dec': 0.006316828613210894,
             'permanence_inc': 0.08130270896950871,
             'permanence_thresh': 0.3287960867800625,
             'predicted_boost': 2,
             'predictive_threshold': 27,
             'segments_per_cell': 10, # 26,    # MODIFIED
             'synapses_per_segment': 60},    # MODIFIED
      'tm_sdrc': {'alpha': 0.0009815495088055376}}

def main(parameters=default_parameters, argv=None, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=int, default=5,
                        help='Number of times to run through the training data.')
    parser.add_argument('--dataset', choices=('states', 'dictionary'), default='states')
    args = parser.parse_args(args = argv)

    # Load data.
    if args.dataset == 'states':
        dataset = state_names
        if verbose:
            print("Dataset is %d state names"%len(dataset))
    elif args.dataset == 'dictionary':
        dataset = read_dictionary()
        dataset = random.sample(dataset, 500)
        if verbose:
            print("Dataset is dictionary words, sample size %d"%len(dataset))

    dataset   = sorted(dataset)
    word_ids  = {word: idx for idx, word in enumerate(sorted(dataset))}
    confusion = np.zeros((len(dataset), len(dataset)))
    if verbose:
        print("Dataset: " + ", ".join('%d) %s'%idx_word for idx_word in enumerate(dataset)))

    # Construct TM.
    diagnostics_alpha = parameters['sp']['boosting_alpha']
    enc = EnumEncoder(**parameters['enc'])
    enc.output_sdr = SDR(enc.output_sdr, average_overlap_alpha = diagnostics_alpha)
    sp = SpatialPooler(
        input_sdr         = enc.output_sdr,
        **parameters['sp'])
    tm = TemporalMemory(
        column_sdr        = sp.columns,
        anomaly_alpha     = diagnostics_alpha,
        **parameters['tm'])
    sdrc = SDRClassifier(steps=[0], **parameters['tm_sdrc'])
    sdrc.compute(-1, [tm.active.size-1],    # Initialize the table.
        classification={"bucketIdx": [len(dataset)-1], "actValue": [len(dataset)-1]},
        learn=True, infer=False)

    def reset():
        enc.output_sdr.zero()
        sp.reset()
        tm.reset()

    # Train.
    if verbose:
        train_cycles = args.time * sum(len(w) for w in dataset)
        print("Training for %d cycles (%d dataset iterations)"%(train_cycles, args.time))
    for i in range(args.time):
        random.shuffle(dataset)
        for word in dataset:
            reset()
            for idx, char in enumerate(word):
                enc.encode(char)
                sp.compute()
                tm.compute()
            lbl = word_ids[word]
            sdrc.compute(tm.age, tm.learning.flat_index,
                classification={"bucketIdx": lbl, "actValue": lbl},
                learn=True, infer=False)

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
