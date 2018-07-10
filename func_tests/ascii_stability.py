#!/usr/bin/python3
"""
Functional Test: ASCII-STABILITY
Written by David McDougall, 2018

This tests that the Stable Spatial Pooler can recognize patterns from a Temporal
Memory.  The system consists of an Encoder, Spatial Pooler & Temporal Memory,
Stable Spatial Pooler, and SDR Classifiers.  The system trains on a random
sequence of words, one character at a time and not separated by spaces.  The
system is then analysed, to determine how well the Stable Spatial Pooler
performs at creating stable representations of words.  Specific things to
analyse:
    1) Shown a single word in isolation which the system has never seen before,
    measure the average overlap throughout the sensation of the word.  Compare
    this to the average overlap response of the system with the stability
    mechanisms disabled.  Also compare this to the overlap between randomly
    generated SDRs. This measures how well the stability mechanism imparts
    stability upon new and inherently unstable inputs.
    2) Shown a single word in isolation which the system has trained on, measure
    the average overlap throughout the sensation.  Compare this to the
    average overlap between different words. This measures the stability within
    an object and between different objects. This determines if the stable
    representations are learned, distinct, and easy to recognise.
    3) Shown a word in numerous different contexts, what is the average overlap
    between different sensation of the word?  Compare this to question #2: the
    average overlap within the word when read in isolation.  This measures how
    well the stability mechanism switches between objects, and if that the
    recognition works with an arbitrary precending context.  
    4) Plot the stability over the course of a sample of input.  Observe if the
    stability dips between words and when ambiguity is resolved.
    5) Use SDR Classifiers to verify that both the Temporal Memory and Stable
    Spatial Pooler convey useful representations of the spatiotemporal patterns
    (aka word).

Optional argument "--practice" alters the training data to make the task easier.
The practice dataset contains each word repeated ??? many times consequtively.
This practive dataset is inserted into the training data, at the halfway point.
"""

# TODO: TM Should receive the TP as context.  The issue is that this breaks it.
# See if enabling the adapt synapses correlated input stuff helps.  Add it to
# the TM!

# TODO: Experiment with dividing TP excitement by the # of connected synapses.

# TODO: Experiment with typos .... Stabiliy should make the system robust
# against them.

import argparse
import random
import itertools
import numpy as np

import sys
sys.path.append('.')
import ascii_tm
from sdr import SDR
from encoders import EnumEncoder
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory
from stable_spatial_pooler import SpatialPooler as StableSpatialPooler
from nupic.algorithms.sdr_classifier import SDRClassifier
from stability import measure_inter_intra_overlap
from synapses import debug as synapses_debug

from ascii_tm import read_dictionary, state_names

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

default_parameters = ascii_tm.default_parameters
default_parameters['tp'] = {
          'active_thresh': 2,
          'mini_columns': 2250,
          'permanence_dec': 0.003675899084079108,
          'permanence_inc': 0.06150975770904488,
          'permanence_thresh': 0.10528865328008735,
          'potential_pool': 1944,
          'segments': 6,
          'sparsity': 0.02510590443564408,
          'stability_rate': 0.06362902461378779}

def main(parameters=default_parameters, argv=None, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=int, default=20,
                        help='Number of times to run through the training data.')
    parser.add_argument('--dataset', choices=('states', 'dictionary'), default='states')
    parser.add_argument('--practice', action='store_true',
        help='Makes the task easier by repeating words.')
    args = parser.parse_args(args = argv)

    # Load data.
    if args.dataset == 'states':
        dataset = state_names
        dataset = [word.replace(' ', '') for word in dataset] # Remove spaces from between the two word states.
        print("Dataset is %d state names"%len(dataset))
    elif args.dataset == 'dictionary':
        dataset = read_dictionary()
        dataset = random.sample(dataset, 200)
        print("Dataset is dictionary words, sample size %d"%len(dataset))

    dataset   = [word.upper() for word in dataset]
    dataset   = sorted(dataset)
    assert(len(dataset) == len(set(dataset)))
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
        # context_sdr       = SDR((parameters['tp']['mini_columns'],)),
        anomaly_alpha     = 1/1000,
        **parameters['tm'])
    tm_sdrc = SDRClassifier(steps=[0])
    tp = StableSpatialPooler(
        input_sdr         = tm.active,
        boosting_alpha    = 1./1000,
        macro_columns     = (1,),
        **parameters['tp'])
    tp_sdrc = SDRClassifier(steps=[0])

    def reset():
        enc.output_sdr.zero()
        sp.reset()
        tm.reset()
        tp.reset()

    def compute(char, learn):
        enc.encode(char)
        sp.compute(learn=learn)
        # tm.context_sdr.flat_index = tp.columns.flat_index
        tm.compute(learn=learn)
        tp.compute(learn=learn) # EXPERIMENT: Try the learning-inputs override.

    t = 0
    def train_word(word, label):
        nonlocal t
        for idx, char in enumerate(word):
            compute(char, learn=True)
        # Process each word before training on the final character.
        tm_sdrc.compute(t, tm.active.flat_index,
            classification={"bucketIdx": label, "actValue": label},
            learn=True, infer=False)
        tp_sdrc.compute(t, tp.columns.flat_index,
            classification={"bucketIdx": label, "actValue": label},
            learn=True, infer=False)
        t += 1

    # TRAIN
    len_dataset = sum(len(w) for w in dataset)
    practice_iterations = 5
    if verbose:
        train_cycles = args.time * len_dataset
        if args.practice:
            train_cycles += len_dataset * practice_iterations
        print("Training for %d cycles"%train_cycles)

    reset()
    for i in range(args.time * len(dataset)):
        word = random.choice(dataset)
        train_word(word, word_ids[word])

        if args.practice and i == args.time * len(dataset) // 2:
            practice_dataset = dataset[:]
            random.shuffle(practice_dataset)
            for word in practice_dataset:
                for attempt in range(practice_iterations):
                    train_word(word, word_ids[word])

    if verbose:
        print("Done training system statistics:")
        print("")
        print("Encoder", enc.output_sdr.statistics())
        print(sp.statistics())
        print(tm.statistics())
        print(tp.statistics())
        print("")

    # TEST
    # Make some new words which the system has never seen before.
    random_words = []
    for word in dataset:
        alphabet    = [chr(ord('A') + i) for i in range(26)]
        random_word = ''.join(random.choice(alphabet) for c in word)
        random_words.append(random_word)
    if verbose:
        print("Novel Words Dataset: " + ', '.join(random_words))
        print("")

    # Measure response to a new random words.
    rand_word_tp_ovlp = 0.
    n_samples         = 0
    for word in random_words:
        reset()
        response = []
        for char in word:
            compute(char, learn = False)
            response.append(SDR(tp.columns))
        for sdr_a, sdr_b in itertools.combinations(response, 2):
            rand_word_tp_ovlp += sdr_a.overlap(sdr_b)
            n_samples += 1
    rand_word_tp_ovlp /= n_samples
    if verbose:
        print("Novel Words, Average Overlap Within Word %g %%"%(100 * rand_word_tp_ovlp))

    # Measure response to a new random word, with the stability mechanism turned
    # off.
    stability_rate = tp.stability_rate
    tp.stability_rate = 1.
    rand_word_tp_ovlp_no_stab = 0.
    for word in random_words:
        reset()
        response = []
        for char in word:
            compute(char, learn = False)
            response.append(SDR(tp.columns))
        for sdr_a, sdr_b in itertools.combinations(response, 2):
            rand_word_tp_ovlp_no_stab += sdr_a.overlap(sdr_b)
    rand_word_tp_ovlp_no_stab /= n_samples
    tp.stability_rate = stability_rate
    if verbose:
        print("Novel Words, No Stability Mechanism, Avg Ovlp Within Word %g %%"%(100 * rand_word_tp_ovlp_no_stab))

    # Compare new word response to that of randomly generated SDRs.
    rand_sdr_ovlp = 0.
    tp_n_active   = int(round(tp.sparsity * tp.columns.size))
    for i in range(n_samples):
        sdr_a = SDR(tp.columns)
        sdr_b = SDR(tp.columns)
        sdr_a.flat_index = np.array(random.sample(range(tp.columns.size), tp_n_active))
        sdr_b.flat_index = np.array(random.sample(range(tp.columns.size), tp_n_active))
        rand_sdr_ovlp += sdr_a.overlap(sdr_b)
    rand_sdr_ovlp /= n_samples
    if verbose:
        print("Random Comparable SDR(n=%d sparsity=%g%%), Average Overlap %g %%"%(
            tp.columns.size,
            100 * tp.sparsity,
            100 * rand_sdr_ovlp),)
        print("")
        print("")

    # Measure response to each word in isolation.
    catagories = {}
    for word in dataset:
        catagories[word] = []
        reset()
        for char in word:
            compute(char, learn = False)
            catagories[word].append(SDR(tp.columns))
    if verbose:
        print("Isolated Word Stability / Distinctiveness:")
    stability, distinctiveness, stability_metric = measure_inter_intra_overlap(catagories, verbose=verbose)
    if verbose:
        print("")

    # Measure response to words in context.  Measure the overlap between the
    # same words in different contexts.  Also check the classifier accuracy.
    min_context  = 3
    catagories   = {}
    tm_accuacy   = 0.
    tp_accuacy   = 0.
    tm_confusion = np.zeros((len(dataset), len(dataset)))
    tp_confusion = np.zeros((len(dataset), len(dataset)))
    n_samples    = 0
    for sample in range(200):
        reset()
        for index in range(20):
            word_index = random.choice(range(len(dataset)))
            word = dataset[word_index]
            if word not in catagories:
                catagories[word] = []
            for char in word:
                compute(char, learn = False)
                if index >= min_context:
                    catagories[word].append(SDR(tp.columns))

            # Check Classifier Accuracy, only if past minimum context words.
            if index < min_context:
                continue
            try:
                tm_inference = tm_sdrc.infer(tm.active.flat_index, None)[0]
            except IndexError:
                tm_inference = np.random.random(size=len(dataset))
            try:
                tp_inference = tp_sdrc.infer(tp.columns.flat_index, None)[0]
            except IndexError:
                tp_inference = np.random.random(size=len(dataset))
            tm_accuacy += word_index == np.argmax(tm_inference)
            tp_accuacy += word_index == np.argmax(tp_inference)
            tm_confusion[word_index] += tm_inference
            tp_confusion[word_index] += tp_inference
            n_samples  += 1
    tm_accuacy /= n_samples
    tp_accuacy /= n_samples
    if verbose:
        print("In-Context Word Stability / Distinctiveness:")
    stability, distinctiveness, stability_metric = measure_inter_intra_overlap(catagories, verbose=verbose)
    if verbose:
        print("")
        print("Number of Samples %g"%n_samples)
        print("Temporal Memory Classifier Accuracy %g %%"%tm_accuacy)
        print("Temporal Pooler Classifier Accuracy %g %%"%tp_accuacy)

    # Display Confusion Matixes
    if verbose:
        conf_matrices = (tm_confusion, tp_confusion,)
        conf_titles   = ('Temporal Memory', 'Temporal Pooler',)
        #
        import matplotlib.pyplot as plt
        plt.figure("Word Recognition Confusion")
        for subplot_idx, matrix_title in enumerate(zip(conf_matrices, conf_titles)):
            matrix, title = matrix_title
            plt.subplot(1, len(conf_matrices), subplot_idx + 1)
            plt.title(title + " Confusion")
            matrix /= np.sum(matrix, axis=1)
            plt.imshow(matrix, interpolation='nearest')
            plt.xlabel('Prediction')
            plt.ylabel('Label')
            for idx, label in enumerate(dataset):
                plt.text(idx, len(dataset) + .5, label, rotation='vertical',
                    horizontalalignment='center', verticalalignment='bottom')
                plt.text(-1.5, idx, label,
                    horizontalalignment='left', verticalalignment='center')

    # Show a sample of input.
    if verbose:
        sentance        = []
        boundries       = []
        anomaly         = []
        stability       = []
        tp_prev_active  = SDR(tp.columns.dimensions)
        n_samples       = 0
        for word_index in range(15):
            word = random.choice(dataset)
            reset()
            for index, char in enumerate(word):
                compute(char, learn = False)
                sentance.append(char)
                if index == 0:
                    boundries.append(n_samples)
                anomaly.append(tm.anomaly)
                stability.append(tp.columns.overlap(tp_prev_active))
                tp_prev_active = SDR(tp.columns)
                n_samples += 1
        plt.figure("ASCII Stability")
        plt.plot(np.arange(n_samples)+.5, anomaly,   'r',
                 np.arange(n_samples)+.5, stability, 'b',)
        for idx, char in enumerate(sentance):
            plt.text(idx + .5, .05, char)
        for x in boundries:
            plt.axvline(x, color='k')
        plt.title("Anomaly is Red, Stability is Blue, Lines are word boundries.")
        plt.show()

    if synapses_debug:
        sp.synapses.check_data_integrity()
        tm.synapses.check_data_integrity()
        tp.synapses.check_data_integrity()
        print("Synapse data structure integrity is OK.")

    # Don't use the temporal memory accuracy, since the temporal memory
    # parameters are stored and optimized in ascii_tm.py.
    score = (stability_metric + (3 * tp_accuacy))
    return score

if __name__ == '__main__':
    main()
