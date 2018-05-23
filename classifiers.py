# Written by David McDougall, 2018

import numpy as np
import bisect 

class SDR_Classifier:
    """Maximum Likelyhood classifier for SDRs."""
    def __init__(self, alpha, input_sdr, num_labels):
        """
        Argument alpha is the small constant used by the exponential moving
        average which tracks input-output co-occurances.
        """
        self.alpha        = alpha
        self.input_sdr    = input_sdr
        self.num_labels   = num_labels
        # Don't initialize to zero, touch every input+output pair.
        self.stats = np.random.uniform(
            0.1 * self.alpha,
            0.2 * self.alpha,
            size=(self.input_sdr.size, self.num_labels))

    def train(self, labels, input_sdr=None):
        """
        Argument labels is array of float, PDF.
        """
        labels = np.array(labels) / np.sum(labels)
        self.input_sdr.assign(input_sdr)
        inputs = self.input_sdr.flat_index
        # Decay.
        self.stats[inputs, :]                *= (1 - self.alpha)
        self.stats[:, np.nonzero(labels)[0]] *= (1 - self.alpha)
        # Update.
        updates = (labels - self.stats[inputs]) * self.alpha
        self.stats[inputs] += updates

    def predict(self, input_sdr=None):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns probability of each catagory in output space.
        """
        self.input_sdr.assign(input_sdr)
        pdf = self.stats[self.input_sdr.flat_index, :]
        pdf = pdf / np.sum(pdf, axis=1, keepdims=True)
        if False:
            # Combine multiple probabilities into single pdf. Product, not
            # summation, to combine probabilities of independant events. The
            # problem with this is if a few unexpected bits turn on it
            # mutliplies the result by zero, and the test dataset is going to
            # have unexpected things in it.  
            return np.product(pdf, axis=0, keepdims=False)
        else:
            # Use summation B/C it works well.
            return np.sum(pdf, axis=0, keepdims=False)

class RandomOutputClassifier:
    """
    This classifier uses the frequency of the trained target outputs to generate
    random predictions.  It is used to get a baseline performance to compare
    against.
    """
    def __init__(self, num_labels):
        self.stats = np.zeros(num_labels)

    def train(self, label):
        label = np.array(label) / np.sum(label)
        self.stats += label

    def predict(self):
        """Returns probability of each catagory in output space."""
        cdf = np.cumsum(self.stats)
        assert(cdf[-1] > 0) # Classifier must be trained before it can make predictions.
        return bisect.bisect(cdf, np.random.random() * cdf[-1])
