"""
This module defines an interface for classifiers, and
implements two classifiers (i.e., NaiveBayes and
Tf-Idf).
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


class Classifier(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self, data):
        pass

    @abstractmethod
    def predict_training(self, folds):
        pass

    @abstractmethod
    def predict(self, data):
        pass


class NaiveBayes(Classifier):
    def __init__(self, data):
        # storing the labels associated with each data point
        self.labels = np.array(data['class'])
        # extracting and storing features from data points
        values = list(data['value'])
        self.vectorizer = CountVectorizer(analyzer = 'word', max_features = 5000)
        self.features = self.vectorizer.fit_transform(values).toarray()
        # training the classifier
        self.gnb = GaussianNB()
        self.gnb.fit(self.features, self.labels)

    def predict_training(self, folds):
        # the module returns the performance of this classifier on the data
        test_gnb = GaussianNB()
        # creating a list of predictions
        # each prediction is a tuple storing the probability of each class
        prediction_list = []
        for itt in range(folds):
            start_index = itt * len(self.features) / folds
            end_index = (itt + 1) * len(self.features) / folds if itt < folds - 1 else len(self.features)
            test_indices = range(start_index,end_index)
            training_features = np.delete(self.features, test_indices, axis=0)
            test_features = self.features[test_indices]
            training_labels = np.delete(self.labels, test_indices, axis=0)
            test_gnb.fit(training_features, training_labels)
            prediction_list.append(test_gnb.predict_proba(test_features))
        return np.concatenate(tuple(prediction_list))

    def predict(self, data):
        # predict the label of new data
        values = list(data['value'])
        features = self.vectorizer.transform(values).toarray()
        return self.gnb.predict_proba(features)


class Tf_Idf(Classifier):
    def __init__(self, data):
        # storing the data for future
        self.labels = np.array(data['class'])
        features_df = data[['value']].copy()
        features_df['length'] = features_df['value'].apply(lambda val: len(val))
        features_df['digit_num'] = features_df['value'].apply(lambda val: sum(char.isdigit() for char in val) / len(val))
        features_df['alpha_num'] = features_df['value'].apply(lambda val: sum(char.isalpha() for char in val) / len(val))
        features_df['space_num'] = features_df['value'].apply(lambda val: sum(char.isspace() for char in val) / len(val))
        self.features = features_df.ix[:,1:].as_matrix()
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(self.features, self.labels)

    def predict_training(self, folds):
        partial_clf = tree.DecisionTreeClassifier()
        prediction_list = []
        for itt in range(folds):
            start_index = itt * len(self.features) / folds
            end_index = (itt + 1) * len(self.features) / folds if itt < folds - 1 else len(self.features)
            test_indices = range(start_index,end_index)
            training_features = np.delete(self.features, test_indices, axis=0)
            test_features = self.features[test_indices]
            training_labels = np.delete(self.labels, test_indices, axis=0)
            partial_clf.fit(training_features, training_labels)
            prediction_list.append(partial_clf.predict_proba(test_features))
        return np.concatenate(tuple(prediction_list))

    def predict(self, data):
        features_df = data[['value']].copy()
        features_df['length'] = features_df['value'].apply(lambda val: len(val))
        features_df['digit_num'] = features_df['value'].apply(lambda val: sum(char.isdigit() for char in val) / len(val))
        features_df['alpha_num'] = features_df['value'].apply(lambda val: sum(char.isalpha() for char in val) / len(val))
        features_df['space_num'] = features_df['value'].apply(lambda val: sum(char.isspace() for char in val) / len(val))
        features = features_df.ix[:,1:].as_matrix()
        return self.clf.predict_proba(features)
