# -*- mode: Python; coding: utf-8 -*-

"""For the purposes of classification, a corpus is defined as a collection
of labeled documents. Such documents might actually represent words, images,
etc.; to the classifier they are merely instances with features."""

from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
from os.path import basename, dirname, split, splitext
from document import Document


class Corpus(object):
    """An abstract collection of documents."""

    __metaclass__ = ABCMeta

    def __init__(self, datafiles, document_class=Document):
        self.documents = []
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    # Act as a mutable container for documents.
    def __len__(self): return len(self.documents)
    def __iter__(self): return iter(self.documents)
    def __getitem__(self, key): return self.documents[key]
    def __setitem__(self, key, value): self.documents[key] = value
    def __delitem__(self, key): del self.documents[key]

    @abstractmethod
    def load(self, datafile, document_class):
        """Make labeled document instances for the data in a file."""
        pass

class PlainTextFiles(Corpus):
    """A corpus contained in a collection of plain-text files."""

    def load(self, datafile, document_class):
        """Make a document from a plain-text datafile. The document is labeled
        using the last component of the datafile's directory."""
        label = split(dirname(datafile))[-1]
        with open(datafile, "r") as file:
            data = file.read()
            self.documents.append(document_class(data, label, datafile))

class PlainTextLines(Corpus):
    """A corpus in which each document is a line in a datafile."""

    def load(self, datafile, document_class):
        """Make a document from each line of a plain text datafile.
        The document is labeled using the datafile name, sans directory
        and extension."""
        label = splitext(basename(datafile))[0]
        with open(datafile, "r") as file:
            for line in file:
                data = line.strip()
                self.documents.append(document_class(data, label, datafile))

class NamesCorpus(PlainTextLines):
    """A collection of names, labeled by gender. See names/README for
    copyright and license."""

    def __init__(self, datafiles="names/*.txt", document_class=Document):
        super(NamesCorpus, self).__init__(datafiles, document_class)

class CSVCorpus(Corpus):
    """A corpus encoded as a comma-separated-value (CSV) file."""

    def load(self, datafile, document_class, encoding="utf-8"):
        """Make a document from each row of a CSV datafile.
        Assumes data, label ordering and UTF-8 encoding."""
        def unicode_csv_reader(csvfile, *args, **kwargs):
            for row in csv_reader(csvfile, *args, **kwargs):
                yield [unicode(cell, encoding) for cell in row]
        with open(datafile, "r") as file:
            for data, label in unicode_csv_reader(file):
                label = label.strip().upper() # canonicalize label
                self.documents.append(document_class(data, label, datafile))

class BlogsCorpus(CSVCorpus):
    """A collection of blog posts, labeled by author gender. See the paper
    "Improving Gender Classification of Blog Authors" by Mukherjee and Liu
    <http://www.cs.uic.edu/~liub/publications/EMNLP-2010-blog-gender.pdf>
    for details and some impressive results."""

    def __init__(self, datafiles="blog-gender-dataset.csv",
                 document_class=Document):
        super(BlogsCorpus, self).__init__(datafiles, document_class)
