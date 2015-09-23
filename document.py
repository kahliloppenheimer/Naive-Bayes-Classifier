from abc import abstractmethod
from collections import Counter, defaultdict
import re

class Document(object):
    """A document completely characterized by its features."""

    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, label=None, source=None):
        self.data = data
        self.label = label
        self.source = source

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())
    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return [self.data]

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return {'iseven Feature': {'isEven': self.data % 2 == 0}}

# Multinomial bag of words representation of a document
class BagOfWords(Document):
    def features(self):
        return {'Bag of Words Feature': Counter([removeNonAlphabetic(x)[:5] for x in self.data.split()])}

class Name(Document):
    def features(self):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        features = defaultdict(Counter)
        name = removeNonAlphabetic(self.data)
        features['first letter Feature'] += Counter({'first letter = ' + name[0]: 1})
        features['last letter Feature'] += Counter({'last letter = ' + name[-1]: 1})
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            count = name.count(letter)
            if count > 0:
                features['letter frequency Feature'] += Counter({'# occurrences of ' + letter: count})
        for letter in {x for x in name}:
            features['letter set feature'] += Counter({'letter appears in word: ' + letter: 1})
        return features

def removeNonAlphabetic(str):
    regex = re.compile('[^a-zA-Z]')
    return regex.sub('', str.lower())



