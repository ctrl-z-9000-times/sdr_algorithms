#!/usr/bin/python3
# Written by David McDougall, 2018

import argparse
import random
import sys
sys.path.append('.')
from temporal_memory import TemporalMemory
from sdr import SDR
from nupic.algorithms.sdr_classifier import SDRClassifier
from encoders import EnumEncoder
import numpy as np

dictionary_file = '/usr/share/dict/american-english'

def read_dictionary():
    with open(dictionary_file) as f:
        return f.read().split()

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

def mutate_word(word):
    """Introduce a random change into the word: delete, swap, repeat, and add
    stray character, do-nothing.  This may raise a ValueError.  """
    if not word:
        return word
    word = list(word)
    choice = random.randrange(6)
    if choice == 0:     # Delete a character
        word.pop(random.randrange(len(word)))
    elif choice == 1:   # Swap two characters
        if len(word) >= 2:
            index = random.randrange(0, len(word) - 1)
            word[index], word[index + 1] = word[index + 1], word[index]
    elif choice == 2:   # Repeat a character
        index = random.randrange(0, len(word))
        word.insert(index, word[index])
    elif choice == 3:   # Insert a stray character
        char = chr(random.randint(ord('A'), ord('Z')))
        word.insert(random.randint(0, len(word)), char)
    elif choice == 4:   # Mutate twice
        word = mutate_word(word)
        word = mutate_word(word)
    elif choice == 5:   # Don't mutate
        pass
    return ''.join(word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=int, default=100,
                        help='Number of times to run through the training data.')
    parser.add_argument('--dataset', choices=('states', 'dictionary'), default='states')
    args = parser.parse_args()

    # Load data.
    if args.dataset == 'states':
        dataset = state_names
        print("Dataset is %d state names"%len(dataset))
    elif args.dataset == 'dictionary':
        dataset = read_dictionary()
        dataset = random.sample(dataset, 1000)
        print("Dataset is dictionary words, sample size %d"%len(dataset))

    dataset  = [word.upper() for word in dataset]
    word_ids = {word: idx for idx, word in enumerate(sorted(dataset))}
    confusion = np.zeros((len(dataset), len(dataset)))

    # Construct TM.
    enc = EnumEncoder(
        size     = 2048,
        sparsity = .04,)
    tm = TemporalMemory(
        column_sdr            = enc.output_sdr,
        add_synapses          = 31,
        cells_per_column      = 32,
        segments_per_cell     = 10,
        synapses_per_segment  = 128,
        permanence_thresh     = .5,
        init_dist             = (0.24, .001),
        permanence_inc        = 0.04,
        permanence_dec        = 0.008,
        mispredict_dec        = 0.001,
        predictive_threshold  = 20,
        learning_threshold    = 13,
        predicted_boost       = 2,
        anomaly_alpha         = 1/1000,)
    sdrc = SDRClassifier(steps=[0])

    # Train.
    print("Training for %d cycles"%(args.time * sum(len(w) for w in dataset)))
    t = 0
    for i in range(args.time):
        random.shuffle(dataset)
        for word in dataset:
            tm.reset()
            for idx, char in enumerate(word):
                enc.encode(char)
                tm.compute()
            lbl = word_ids[word]
            sdrc.compute(t, tm.active.flat_index,
                classification={"bucketIdx": lbl, "actValue": lbl},
                learn=True, infer=False)
            t += 1

    print(tm.statistics())

    # Test.
    score = 0
    score_samples = 0
    for word in dataset:
        tm.reset()
        for idx, char in enumerate(word):
            enc.encode(char)
            tm.compute(learn=False)

        inference = sdrc.infer(tm.active.flat_index, None)
        lbl = word_ids[word]
        if lbl == np.argmax(inference[0]):
            score += 1
        score_samples += 1
        confusion[lbl] += inference[0]
    print("Score:", 100 * score / score_samples, '%')

    tm.synapses.check_data_integrity()
    print("Synapse data structure integrity is OK.")

    import matplotlib.pyplot as plt
    plt.figure('Confusion Matrix')
    plt.imshow(confusion, interpolation='nearest')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
