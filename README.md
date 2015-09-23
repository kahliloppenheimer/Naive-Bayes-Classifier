# Naive-Bayes-Classifier

## About
This is my implementation of a Naive Bayes Classifier for natural language in Python. For my testing, I used a blog dataset annotated with the genders of the authors and achieved 71% accuracy during classifier evaluation.

## How to classify a new type of document
All you need to do is add a subclass of Document that corresponds to the new type of document you'd like to classify. This subclass just needs to implement a getFeatures() method which returns a map of feature names to features, i.e.

{'bag of words Feature' -> 
    {'the': 50,
     'and': 75,
     ...
    },
 'Average word length feature' ->
    {'average word length' -> 8},
 ...
}

The classifier is generic enough that it will properly train and classify any documents once you've provided this behavior.
