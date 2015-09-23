# -*- mode: Python; coding: utf-8 -*-
from __future__ import division
from collections import defaultdict, Counter
from numbers import Number
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL

import math
from time import sleep

from classifier import Classifier

class NaiveBayes(Classifier):

    # Values used for laplace smoothing
    ALPHA = 1
    ALPHA_SCALE = 2
    # Real between 0 and 1 that gets multiplied by the probability of each
    # feature component present in the classified document, but not the model
    UNKNOWN_PENALTY = .1

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)
        # Model maps class -> featureName -> featureVec
        # I.e. {
        #       male -> {'bag of words' -> {the: 4, and: 6, ...}, # sentences -> { # = 5 }, etc.,},
        #       female -> {'# noun phrases' -> { # = 6 }, etc.}
        #      }
        self._model = defaultdict(defaultDictOfCounters)
        # Keeps track of total # of values per feature per class across the entire corpus
        # (i.e. denominator when normalizing probability)
        # I.e. {
        #       male -> {bag of words -> 500 words, # sentences -> 100 sentences, etc.,},
        #       female -> {'# noun phrases' -> 36}
        #      }
        self.countPerFeature = defaultdict(Counter)

        # Keeps track of overall counts of each label
        self.priorCount = Counter()

    def get_model(self): return self._model
    def set_model(self, model): self._model = model
    model = property(get_model, set_model)

    # Trains this classifier on the given set of instances
    def train(self, instances):
        self.count(instances)
        self.smooth(self.laplaceSmooth)

    # Classifies an instance given the trained model
    def classify(self, instance):
        features = instance.features()
        # Calculate probabilities for each class prediction
        labelPredictions = {}
        for label in self._model:
            # Running sum of logged probabilities
            logProb = 0
            for feature, featureVec in features.iteritems():
                # Count known elements
                for featureComp, count in featureVec.iteritems():
                    nextProb = self._model[label][feature][featureComp]
                    # Add logged probability normally if it was present in model, else penalize
                    if(nextProb > 0):
                        logProb += count * math.log(nextProb)
                    else:
                        logProb += count * math.log(self.UNKNOWN_PENALTY * self.laplaceSmooth(1, self.countPerFeature[label][feature]))
            # Factor in prior probability of label
            logProb += math.log(self.priorCount[label] / sum(self.priorCount.values()))

            labelPredictions[label] = logProb

        return max(labelPredictions, key=labelPredictions.get)

    # Counts occurrences of features within each instance. Features associated
    # with "truthy" non-numeric values are counted as 1. Numeric feature values
    # are simply summed up, thus the distinction between Bernouli/Multinomial
    # is handled by which numbers are associated with each feature
    def count(self, instances):
        # Reset prior counts, feature counts, and model counts
        self.priorCount.clear()
        self.countPerFeature.clear()
        for label, feature in self._model.iteritems():
            for featureVec in feature:
                featureVec.clear()

        # Perform counting of feature occurrences in training data
        for instance in instances:
            self.priorCount[instance.label] += 1
            for feature, featureVec in instance.features().iteritems():
                for featureComp, count in featureVec.iteritems():
                    # Handle case of non-numeric feature value (like True)
                    if(count and not isinstance(count, Number)):
                        self._model[instance.label][feature][featureComp] += 1
                        self.countPerFeature[instance.label][feature] += 1
                    # Handle expected case of numeric count
                    elif isinstance(count, Number):
                        self._model[instance.label][feature][featureComp] += count
                        self.countPerFeature[instance.label][feature] += count

    # Iterates over the counted model and normalizes/smooths each value according to a
    # passed smoothing function. The function takes two parameters:
    # 1) the count for a particular feature
    # 2) the summed count of all features for the instance's label
    def smooth(self, smoothingFunc):
    # Normalize results via laplace smoothing
        for label, features in self._model.iteritems():
            for feature, featureVec in features.iteritems():
                for featureComp, count in featureVec.iteritems():
                    # print "smoothing ", count, " / ", self.priorCount[label], " to ", smoothingFunc(count, self.priorCount[label])
                    self._model[label][feature][featureComp] = smoothingFunc(count, self.countPerFeature[label][feature])

    # Smooths the probability estimate using laplace smoothing
    def laplaceSmooth(self, elemCount, totalCount):
        return (elemCount + self.ALPHA) / (totalCount + (self.ALPHA * self.ALPHA_SCALE))

    # Computes the prior and feature counts according to the current model.
    # Especially useful when saving/loading the model via Pickel
    def initPriorAndFeatureCounts(self):
        self.priorCount = Counter()
        self.countPerFeature = defaultdict(Counter)
        for label in self._model:
            for feature, featureVec in self._model[label].iteritems():
                self.priorCount[label] += 1
                for featureComp, count in featureVec.iteritems():
                    self.countPerFeature[label][feature] += count if isinstance(count, Number) else 1

    # Saves the model, prior count, and count per feature
    def save(self, file):
        """Save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.save(file)
        else:
            dump([self.model, self.priorCount, self.countPerFeature], file, HIGHEST_PICKLE_PROTOCOL)

    # Loads a file containing the model, prior count, and counts per feature
    def load(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.load(file)
        else:
            loaded = load(file)
            self.model = loaded[0]
            self.priorCount = loaded[1]
            self.countPerFeature = loaded[2]


# Module level function to be used as default function for another defaultdict.
# This is because defaultdicts cannot be serialized with pickle when using
# lambda expressions
def defaultDictOfCounters():
    return defaultdict(Counter)
