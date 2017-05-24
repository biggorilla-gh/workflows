"""
Implement classifier for FlexMatcher.

This module defines an interface for classifiers, and
implements two classifiers (i.e., NaiveBayes and
Tf-Idf).

Todo:
    * Implement more relevant classifiers.
    * Implement simple rules (e.g., does data match a phone number?).
    * Shuffle data before k-fold cutting in predict_training.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


class Classifier(object):

    """Define classifier interface for FlexMatcher."""
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self, data):
        """Train based on the input training data."""
        pass

    @abstractmethod
    def predict_training(self, folds):
        """Predict the training data (using k-fold cross validation)."""
        pass

    @abstractmethod
    def predict(self, data):
        """Predict for unseen data."""
        pass


class NGramClassifier(Classifier):

    """Classify data-points using counts of n-gram sequence of words or chars.

    The NGramClassifier uses n-grams of words or characters (based on user
    preference) and extracts count features or binary features (based on user
    preference) to train a classifier. It uses the Gaussian naive Bayes
    classifier as its training model.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        features (ndarray): Matrix storing the extracting features.
        vectorizer (object): Vectorizer for transforming text to features. It
        will be either of type CountVectorizer or HashingVectorizer.
        clf (DecisionTreeClassifier): The classifier instance.
        num_classes (int): Number of classes/columns to match to
    """

    def __init__(self, data, ngram_range=(1,1), analyzer='word', count=True):
        """Extracts features and labels from the data and fits a model.

        Args:
            data (dataframe): Training data (values and their correct column).
            ngram_range (tuple): Pair of ints specifying the range of ngrams.
            analyzer (string): Determines what type of analyzer to be used.
            Setting it to 'word' will consider each word as a unit of language
            and 'char' will consider each character as a unit of language.
            count (boolean): Determines if features are counts of n-grams
            versus a binary value encoding if the n-gram is present or not.
        """
        self.labels = np.array(data['class'])
        self.num_classes = len(data['class'].unique())
        values = list(data['value'])
        # checking what type of vectorize to create
        if count:
            self.vectorizer = CountVectorizer(analyzer = analyzer,
                                              ngram_range = ngram_range)
        else:
            self.vectorizer = HashingVectorizer(analyzer = analyzer,
                                                ngram_range = ngram_range)
        self.features = self.vectorizer.fit_transform(values).toarray()
        # training the classifier
        self.gnb = GaussianNB()
        self.gnb.fit(self.features, self.labels)

    def predict_training(self, folds=5):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        partial_clf = GaussianNB()
        prediction = np.zeros((len(self.features), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.features, self.labels):
            # prepare the training and test data
            training_features = self.features[train_index]
            test_features = self.features[test_index]
            training_labels = self.labels[train_index]
            # fitting the model and predicting
            partial_clf.fit(training_features, training_labels)
            prediction[test_index] = partial_clf.predict_proba(test_features)
        return prediction

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        values = list(data['value'])
        features = self.vectorizer.transform(values).toarray()
        return self.gnb.predict_proba(features)


class CharDistClassifier(Classifier):

    """Classify the data-point using counts of character types in the data.

    The CharDistClassifier extracts 7 simple features: number of
    white-space, digit, and alphabetical characters as well as their percentage
    and the total number of characters. Then it trains a decision tree on top
    of these features.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        features (ndarray): Matrix storing the extracting features.
        clf (DecisionTreeClassifier): The classifier instance.
        num_classes (int): Number of classes/columns to match to
    """

    def __init__(self, data):
        """Extracts features and labels from the data and fits a model.

        Args:
            data (dataframe): Training data (values and their correct column).
        """
        self.labels = np.array(data['class'])
        self.num_classes = len(data['class'].unique())
        # populating the features dataframe
        feat_df = data[['value']].copy()
        feat_df['length'] = feat_df['value'].apply(lambda val: len(val))
        feat_df['digit_frac'] = feat_df['value'].apply(
            lambda val: sum(char.isdigit() for char in val) / len(val))
        feat_df['digit_num'] = feat_df['value'].apply(
            lambda val: sum(char.isdigit() for char in val))
        feat_df['alpha_frac'] = feat_df['value'].apply(
            lambda val: sum(char.isalpha() for char in val) / len(val))
        feat_df['alpha_num'] = feat_df['value'].apply(
            lambda val: sum(char.isalpha() for char in val))
        feat_df['space_frac'] = feat_df['value'].apply(
            lambda val: sum(char.isspace() for char in val) / len(val))
        feat_df['space_num'] = feat_df['value'].apply(
            lambda val: sum(char.isspace() for char in val))
        self.features = feat_df.ix[:,1:].as_matrix()
        # training the classifier
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(self.features, self.labels)

    def predict_training(self, folds=5):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        partial_clf = tree.DecisionTreeClassifier()
        prediction = np.zeros((len(self.features), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.features, self.labels):
            # prepare the training and test data
            training_features = self.features[train_index]
            test_features = self.features[test_index]
            training_labels = self.labels[train_index]
            # fitting the model and predicting
            partial_clf.fit(training_features, training_labels)
            prediction[test_index] = partial_clf.predict_proba(test_features)
        return prediction

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        feat_df = data[['value']].copy()
        feat_df['length'] = feat_df['value'].apply(lambda val: len(val))
        feat_df['digit_frac'] = feat_df['value'].apply(
            lambda val: sum(char.isdigit() for char in val) / len(val))
        feat_df['digit_num'] = feat_df['value'].apply(
            lambda val: sum(char.isdigit() for char in val))
        feat_df['alpha_frac'] = feat_df['value'].apply(
            lambda val: sum(char.isalpha() for char in val) / len(val))
        feat_df['alpha_num'] = feat_df['value'].apply(
            lambda val: sum(char.isalpha() for char in val))
        feat_df['space_frac'] = feat_df['value'].apply(
            lambda val: sum(char.isspace() for char in val) / len(val))
        feat_df['space_num'] = feat_df['value'].apply(
            lambda val: sum(char.isspace() for char in val))
        features = feat_df.ix[:,1:].as_matrix()
        return self.clf.predict_proba(features)
