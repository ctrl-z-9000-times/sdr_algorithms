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
    stability dips between words and when ambiguity is resolved.  Use the  
    "--learned_stability" flag to disabled the stability mechanism during
    testing and see what was learned during training.
    5) Use SDR Classifiers to verify that both the Temporal Memory and Stable
    Spatial Pooler convey useful representations of the spatiotemporal patterns
    (aka word).

Optional argument "--practice" alters the training data to make the task easier.
The practice argument is an integer, the number of times to practice each word.
Practice words are inserted into the middle of the training data, at the halfway
point.  The practice words are repeated several times consequtively.

Optional argument "--typo" introduces misspellings into the test dataset.
"""

import argparse
import random
import itertools
import os
import re
import numpy as np

import sys
sys.path.append('.')
from ascii_tm import read_dictionary, state_names
from sdr import SDR
from encoders import EnumEncoder
from spatial_pooler import SpatialPooler, StableSpatialPooler
from temporal_memory import TemporalMemory
from nupic.algorithms.sdr_classifier import SDRClassifier
from stability import measure_inter_intra_overlap
from synapses import debug as synapses_debug

def overlap_stability_weighted(timeseries):
    """
    Argument timeseries is a list of SDRs.
    Returns a list of values in range [0, 1]
    """
    assert(all(isinstance(sdr, SDR) for sdr in timeseries))
    assert(all(sdr.dimensions == timeseries[0].dimensions for sdr in timeseries))
    # Make table containing how many cycles each SDR bit has been active for,
    # and another table for how long each bit will be active for in the future.
    # Both tables also contain the current activations at each timestep.
    forward  = np.zeros((len(timeseries), timeseries[0].size), dtype=np.int)
    backward = np.zeros((len(timeseries), timeseries[0].size), dtype=np.int)
    # Forward pass calculates prior activation history.
    for time, sdr in enumerate(timeseries):
        forward[time][sdr.flat_index] += 1
        if time > 0:
            forward[time][sdr.flat_index] += forward[time - 1][sdr.flat_index]
    # Backwards pass calculates future activation history.
    for time, sdr in reversed(list(enumerate(timeseries))):
        backward[time][sdr.flat_index] += 1
        if time + 1 < len(timeseries):
            backward[time][sdr.flat_index] += backward[time + 1][sdr.flat_index]
    # Calculate the overlap between each timestep.  Each bit is weighted by how
    # long it's active for, both in the past and future.
    overlaps = [0.]  # At time=0, overlap is 0.
    for time in range(1, len(timeseries)):
        stable_weight = forward[time - 1] + backward[time]
        denominator   = np.sum(stable_weight)
        sdr_prior     = timeseries[time - 1].flat_index
        sdr_post      = timeseries[time].flat_index
        ovlp_bits     = np.intersect1d(sdr_prior, sdr_post)
        numerator     = np.sum(stable_weight[ovlp_bits])
        ovlp_weighted = numerator / float(denominator)
        overlaps.append(ovlp_weighted)
    return overlaps

def read_gutenberg(length=1e6, verbose=True):
    """
    Converts to all uppercase.
    Removes all whitespace.
    Converts all whitespace (and underscores) to space characters.
    """
    # Make a list of all available texts.
    data_path = 'func_tests/project_gutenberg'
    books     = os.listdir(data_path)
    random.shuffle(books)   # Pick books at random.

    dataset = []
    while len(dataset) < length:
        try:
            selected_book = books.pop()
        except IndexError:
            break
        if verbose:
            print("Reading", selected_book)
        selected_book_path = os.path.join(data_path, selected_book)
        with open(selected_book_path) as file:
            text = file.read()
        text = text.upper()
        text = re.split("[^A-Z]", text)
        text = (word for word in text if word)
        dataset.extend(text)
    return dataset[ : int(length)]

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

default_parameters = {
    'enc': {'size': 3166, 'sparsity': 0.04226182627266593},
    'sp': {'boosting_alpha': 0.004997591476248401,
         'mini_columns': 1032,
         'permanence_dec': 0.0009668770173435322,
         'permanence_inc': 0.045971813340395094,
         'permanence_thresh': 0.008579047726599218,
         'potential_pool': 1809,
         'sparsity': 0.0923451538565662},
    'tm': {'add_synapses': 11,
         'cells_per_column': 20,
         'init_dist': (0.5598232301060806, 0.006168285991932362),
         'learning_threshold': 4,
         'mispredict_dec': 0.005727263850540909,
         'permanence_dec': 0.0047410936432661895,
         'permanence_inc': 0.08467244215602884,
         'permanence_thresh': 0.5614494062462105,
         'predicted_boost': 1,
         'predictive_threshold': 9,
         'segments_per_cell': 22,
         'synapses_per_segment': 50},
    'tm_sdrc': {'alpha': 0.0003969982399689994},
    'tp': {'active_thresh': -3,
         'boosting_alpha': 0.0003269537573903356,
         'mini_columns': 3000,
         'permanence_dec': 0.001182906466338482,
         'permanence_inc': 0.028816806683464806,
         'permanence_thresh': 0.04331108582593191,
         'potential_pool': 6604,
         'segments': 4,
         'sparsity': 0.00730688371636284,
         'stability_rate': 0.27023961606581576},
    'tp_sdrc': {'alpha': 0.0006894941776977128},
    'tp_nz_value' : 1,}

if True:
    default_parameters['tm']['cells_per_column']  = 32
    default_parameters['tp']['potential_pool']    *= 32/20
    # default_parameters['tm']['segments_per_cell'] = 26
    # default_parameters['tp']['mini_columns']      = 4000

def main(parameters=default_parameters, argv=None, verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', type=int, default=20,
                        help='Number of times to run through the training data.')
    parser.add_argument('--dataset', choices=('states', 'dictionary', 'gutenberg'),
        default='states')
    parser.add_argument('--words', type=int, default=200,
        help='Number of words to use.')
    parser.add_argument('--typo', type=float, default=0.,
        help='Misspell words, percentage [0-1], default 0.')
    parser.add_argument('--practice', type=int, default=0,
        help='Makes the task easier by repeating words.')
    parser.add_argument('--learned_stability', action='store_true',
        help='Disable the stability mechanism during tests.')
    args = parser.parse_args(args = argv)

    if verbose:
        print("Parameters = ")
        import pprint
        pprint.pprint(parameters)
        print("")

    # Load dataset.  The dataset consists of three variables:
    # 1) training_data is a list of words.
    # 2) testing_data is a list of words.
    # 3) dataset is dictionary of word -> identifier pairs.
    if args.dataset == 'states':
        # Remove spaces from between the two word states names.
        dataset       = [word.replace(' ', '') for word in state_names]
        training_data = dataset * args.time
        testing_data  = dataset * 5
        random.shuffle(training_data)
        random.shuffle(testing_data)
        if verbose:
            print("Dataset is %d state names."%len(dataset))
    elif args.dataset == 'dictionary':
        dataset       = read_dictionary()
        dataset       = random.sample(dataset, args.words)
        training_data = dataset * args.time
        testing_data  = dataset * 5
        random.shuffle(training_data)
        random.shuffle(testing_data)
        if verbose:
            print("Dataset is %d dictionary words."%len(dataset))
    elif args.dataset == 'gutenberg':
        text          = read_gutenberg(args.time)
        split         = int(.80 * len(text))    # Fraction of data to train on.
        training_data = text[ : split]
        testing_data  = text[split : ]
        # Put the most common words into the dataset to be trained & tested on.
        histogram     = {}
        for word in training_data:
            if word not in histogram:
                histogram[word] = 0
            histogram[word] += 1
        histogram.pop('S', None)    # Remove apostrophy 'S'.
        dataset = sorted(histogram, key = lambda word: histogram[word])
        dataset = dataset[ -args.words : ]
        if verbose:
            print("Dataset is %d words from Project Gutenberg."%len(dataset))
            unique_train = len(set(training_data))
            unique_test  = len(set(testing_data))
            print("Unique words in training data %d, testing data %d"%(unique_train, unique_test))

    dataset = {word: idx for idx, word in enumerate(sorted(set(dataset)))}
    if verbose:
        print("Training data %d words, %g%% dataset coverage."%(
            len(training_data),
            100. * sum(1 for w in training_data if w in dataset) / len(dataset)))
        print("Testing data %d words, %g%% dataset coverage."%(
            len(testing_data),
            100. * sum(1 for w in testing_data if w in dataset) / len(dataset)))
        print("Dataset: " + ", ".join('%d) %s'%(dataset[word], word) for word in sorted(dataset)))

    if args.practice:
        insertion_point  = int(len(training_data) / 2)
        practice_dataset = list(dataset)
        random.shuffle(practice_dataset)
        for word in practice_dataset:
            for attempt in range(args.practice):
                training_data.insert(insertion_point, word)

    # Construct TM.
    diagnostics_alpha = parameters['sp']['boosting_alpha']
    enc = EnumEncoder(**parameters['enc'])
    enc.output_sdr = SDR(enc.output_sdr, average_overlap_alpha = diagnostics_alpha)
    sp = SpatialPooler(
        input_sdr         = enc.output_sdr,
        **parameters['sp'])
    tm = TemporalMemory(
        column_sdr        = sp.columns,
        context_sdr       = SDR((parameters['tp']['mini_columns'],)),
        anomaly_alpha     = diagnostics_alpha,
        **parameters['tm'])
    tm_sdrc = SDRClassifier(steps=[0], **parameters['tm_sdrc'])
    tm_sdrc.compute(-1, [tm.active.size-1],    # Initialize the SDRCs internal table.
        classification={"bucketIdx": [len(dataset)-1], "actValue": [len(dataset)-1]},
        learn=True, infer=False)
    tp = StableSpatialPooler(
        input_sdr         = tm.active,
        macro_columns     = (1,),
        **parameters['tp'])
    tp_sdrc = SDRClassifier(steps=[0], **parameters['tp_sdrc'])
    tp_sdrc.compute(-1, [tp.columns.size-1],    # Initialize the SDRCs internal table.
        classification={"bucketIdx": [len(dataset)-1], "actValue": [len(dataset)-1]},
        learn=True, infer=False)

    def reset():
        enc.output_sdr.zero()
        sp.reset()
        tm.reset()
        tp.reset()

    def compute(char, learn):
        enc.encode(char)
        sp.compute(learn=learn)
        tm.context_sdr.flat_index = tp.columns.flat_index
        tm.context_sdr.nz_values.fill(parameters['tp_nz_value'])
        tm.compute(learn=learn)
        tp.compute(learn=learn,
            input_learning_sdr = tm.learning,)

    # TRAIN
    if verbose:
        train_cycles = sum(len(w) for w in training_data)
        iterations   = len(training_data) / len(dataset)
        print("Training for %d cycles (%d dataset iterations)"%(train_cycles, iterations))

    reset()
    for word in training_data:
        for idx, char in enumerate(word):
            compute(char, learn=True)
        # Process each word before training on the final character.
        try:
            label = dataset[word]
        except KeyError:
            continue
        if len(tm.learning):
            tm_sdrc.compute(tm.age, tm.learning.flat_index,
                classification={"bucketIdx": label, "actValue": label},
                learn=True, infer=False)
        if len(tp.columns):
            tp_sdrc.compute(tp.age, tp.columns.flat_index,
                classification={"bucketIdx": label, "actValue": label},
                learn=True, infer=False)

    if verbose:
        print("Done training.  System statistics:")
        print("")
        print("Encoder", enc.output_sdr.statistics())
        print(sp.statistics())
        print(tm.statistics())
        print(tp.statistics())
        print("")

    # TEST
    # Make some new words which the system has never seen before.
    if verbose:
        random_words = []
        for word in dataset:
            alphabet    = [chr(ord('A') + i) for i in range(26)]
            random_word = ''.join(random.choice(alphabet) for c in word)
            random_words.append(random_word)
        print("Novel Words Dataset: " + ', '.join(random_words))
        print("")

        # Measure response to new random words.
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
        print("Novel Words, Average Overlap Within Word %g %%"%(100 * rand_word_tp_ovlp))

        # Measure response to new random words, with the stability mechanism
        # turned off.
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
        print("Novel Words, No Stability Mechanism, Avg Ovlp Within Word %g %%"%(100 * rand_word_tp_ovlp_no_stab))

        # Compare new word response to that of randomly generated SDRs.
        rand_sdr_ovlp = 0.
        tp_n_active   = len(tp.columns)
        for i in range(n_samples):
            sdr_a = SDR(tp.columns)
            sdr_b = SDR(tp.columns)
            sdr_a.flat_index = np.array(random.sample(range(tp.columns.size), tp_n_active))
            sdr_b.flat_index = np.array(random.sample(range(tp.columns.size), tp_n_active))
            rand_sdr_ovlp += sdr_a.overlap(sdr_b)
        rand_sdr_ovlp /= n_samples
        print("Random Comparable SDR(n=%d sparsity=%g%%), Average Overlap %g %%"%(
            tp.columns.size,
            100 * tp_n_active / tp.columns.size,
            100 * rand_sdr_ovlp),)
        print("")

    if args.learned_stability:
        tp.stability_rate = 1
        if verbose:
            print("")
            print("Disabled Stability Mechanism...")
            print("")

    # Measure response to each word in isolation.
    if verbose:
        catagories   = {word : [] for word in dataset}
        tm_accuacy   = 0.
        tp_accuacy   = 0.
        n_samples    = 0
        for word, word_id in dataset.items():
            reset()
            for char in word:
                compute(char, learn = False)
                catagories[word].append(SDR(tp.columns))
            try:
                tm_inference = tm_sdrc.infer(tm.active.flat_index, None)[0]
            except IndexError:
                tm_inference = np.random.random(size=len(dataset))
            try:
                tp_inference = tp_sdrc.infer(tp.columns.flat_index, None)[0]
            except IndexError:
                tp_inference = np.random.random(size=len(dataset))
            tm_accuacy += word_id == np.argmax(tm_inference)
            tp_accuacy += word_id == np.argmax(tp_inference)
            n_samples  += 1
        tm_accuacy /= n_samples
        tp_accuacy /= n_samples
        print("")
        print("Isolated Word Stability / Distinctiveness:")
        stability, distinctiveness, stability_metric = measure_inter_intra_overlap(catagories, verbose=verbose)
        print("Temporal Memory Classifier Accuracy %g %% (%d samples)"%(100 * tm_accuacy, n_samples))
        print("Temporal Pooler Classifier Accuracy %g %% (%d samples)"%(100 * tp_accuacy, n_samples))
        print("")

    # Measure response to words in context.  Measure the overlap between the
    # same words in different contexts.  Also check the classifier accuracy.
    catagories   = {word : [] for word in dataset}
    tm_accuacy   = 0.
    tp_accuacy   = 0.
    tm_confusion = np.zeros((len(dataset), len(dataset)))
    tp_confusion = np.zeros((len(dataset), len(dataset)))
    n_samples    = 0
    reset()
    for word in testing_data:
        if random.random() < args.typo:
            mutated_word = mutate_word(word)
        else:
            mutated_word = word

        for char in mutated_word:
            compute(char, learn = False)
            if word in catagories:
                catagories[word].append(SDR(tp.columns))

        # Check Classifier Accuracy.
        try:
            word_id = dataset[word]
        except KeyError:
            continue
        try:
            tm_inference = tm_sdrc.infer(tm.active.flat_index, None)[0]
        except IndexError:
            tm_inference = np.random.random(size=len(dataset))
        try:
            tp_inference = tp_sdrc.infer(tp.columns.flat_index, None)[0]
        except IndexError:
            tp_inference = np.random.random(size=len(dataset))
        tm_accuacy += word_id == np.argmax(tm_inference)
        tp_accuacy += word_id == np.argmax(tp_inference)
        n_samples  += 1
        tm_confusion[word_id] += tm_inference / np.sum(tm_inference)
        tp_confusion[word_id] += tp_inference / np.sum(tp_inference)
    tm_accuacy /= n_samples
    tp_accuacy /= n_samples
    if verbose:
        print("")
        print("In-Context Word Stability / Distinctiveness:")
    stability, distinctiveness, stability_metric = measure_inter_intra_overlap(catagories, verbose=verbose)
    if verbose:
        print("Temporal Memory Classifier Accuracy %g %% (%d samples)"%(100 * tm_accuacy, n_samples))
        print("Temporal Pooler Classifier Accuracy %g %% (%d samples)"%(100 * tp_accuacy, n_samples))

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
            matrix /= np.sum(matrix, axis=0)
            plt.imshow(matrix, interpolation='nearest')
            plt.xlabel('Prediction')
            plt.ylabel('Label')
            for label, idx in dataset.items():
                plt.text(idx, len(dataset) + .5, label, rotation='vertical',
                    horizontalalignment='center', verticalalignment='bottom')
                plt.text(-1.5, idx, label,
                    horizontalalignment='left', verticalalignment='center')

    # Show a sample of input.
    if verbose:
        sentance        = []
        boundries       = []
        anomaly_hist    = []
        stability_hist  = []
        tp_active_hist  = []
        tp_prev_active  = SDR(tp.columns.dimensions)
        n_samples       = 0
        sample_data     = testing_data[ : 100]
        reset()
        for word in sample_data:
            if random.random() < args.typo:
                mutated_word = mutate_word(word)
            else:
                mutated_word = word

            for index, char in enumerate(mutated_word):
                compute(char, learn = False)
                sentance.append(char)
                if index == 0:
                    boundries.append(n_samples)
                anomaly_hist.append(tm.anomaly)
                tp_active_hist.append(SDR(tp.columns))
                stability_hist.append(tp.columns.overlap(tp_prev_active))
                tp_prev_active = SDR(tp.columns)
                n_samples += 1

        plt.figure("ASCII Stability")
        stability_weighted = overlap_stability_weighted(tp_active_hist)
        plt.plot(
                 # np.arange(n_samples)+.5, anomaly_hist,   'ro',
                 # np.arange(n_samples)+.5, stability_hist, 'b-',
                 np.arange(n_samples)+.5, stability_weighted, 'b-',)
        for idx, char in enumerate(sentance):
            plt.text(idx + .5, -.04, char, horizontalalignment='center')
        for x in boundries:
            plt.axvline(x, color='k')
        figure_title = "ASCII Stability"
        if args.learned_stability:
            figure_title += " - Stability Mechanism Disabled."
        figure_title += "\nLines are word boundries."
        plt.title(figure_title)
        plt.show()

    if synapses_debug:
        sp.synapses.check_data_integrity()
        tm.synapses.check_data_integrity()
        tp.synapses.check_data_integrity()
        print("Synapse data structure integrity is OK.")

    score = (stability * tm_accuacy * tp_accuacy)
    if verbose:
        print("Score: %g"%score)
    return score

if __name__ == '__main__':
    main()
