from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import flexmatcher.classify as clf
from sklearn import linear_model
import numpy as np
import pandas

class FlexMatcher:

    """Match a given schema to the mediated schema.

    The FlexMatcher learns to match an input schema to a mediated schema.
    The class considers panda dataframes as databases and their column names as
    the schema. FlexMatcher learn to do schema matching by training on
    instances of dataframes and how their columns are matched against the
    mediated schema.

    Attributes:
        training_data (dataframe): Dataframe where with 3 columns. The name of
            the column in the schema, the value under that column and the name
            of the column in the mediated schema it was mapped to.
        classifier_list (list): List of classifiers used in the training.
        prediction_list (list): List of predictions on the training data
            produced by each classifier.
        weights (ndarray): A matrix where cell (i,j) captures how good the j-th
            classifier is at predicting if a column should match the i-th
            column (where columns are sorted by name) in the mediated schema.
        columns ()
    """

    def __init__(self):
        pass

    def create_training_data(self, dataframes, mappings, sample_size=100):
        """Transform dataframes and mappings into training data.

        The method uses the names of columns as well as the data under each
        column as its training data. It also replaces missing values with 'NA'.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
        """
        training_data_list = []
        for (datafr, mapping) in zip(dataframes, mappings):
            sampled_rows = datafr.sample(min(sample_size, datafr.shape[0]))
            sampled_data = pandas.melt(sampled_rows)
            sampled_data.columns = ['name', 'value']
            sampled_data['class'] = \
                sampled_data.apply(lambda row: mapping[row['name']], axis=1)
            training_data_list.append(sampled_data)
        training_data = pandas.concat(training_data_list, ignore_index=True)
        self.training_data = training_data.fillna('NA')
        self.columns = list(set.union(*[set(x.values()) for x in mappings]))

    def train(self):
        """Train each classifier and the meta-classifier."""
        classifierA = clf.NaiveBayes(self.training_data)
        classifierB = clf.Tf_Idf(self.training_data)
        self.classifier_list = [classifierA, classifierB]
        self.prediction_list = \
            [x.predict_training() for x in self.classifier_list]
        self.train_meta_learner()

    def train_meta_learner(self):
        """Train the meta-classifier.

        The data used for training the meta-classifier is the probability of
        assigning each point to each column (or class) by each classifier. The
        learned weights suggest how good each classifier is at predicting a
        particular class."""
        coeff_list = []
        for class_ind, class_name in enumerate(self.columns):
            # preparing the dataset for linear regression
            regression_data = self.training_data[['class']].copy()
            regression_data['is_class'] = \
                np.where(self.training_data['class'] == class_name, True, False)
            # adding the prediction probability from classifiers
            for classifier_ind, prediction in enumerate(self.prediction_list):
                regression_data['classifer' + str(classifier_ind)] = \
                    prediction[:,class_ind]
            # setting up the linear regression
            stacker = linear_model.LinearRegression()
            stacker.fit(regression_data.iloc[:,2:], regression_data['is_class'])
            coeff_list.append(stacker.coef_.reshape(1,-1))
        self.weights = np.concatenate(tuple(coeff_list))

    def make_prediction(self, data):
        """Map the schema of a given dataframe to the column of mediated schema.

        The procedure runs each classifier and then uses the weights (learned
        by the meta-trainer) to combine the prediction of each classifier.
        """
        data = data.fillna('NA')
        # predicting each column
        predicted_mapping = {}
        for column in list(data):
            column_dat = data[[column]]
            column_dat.columns = ['value']
            scores = np.zeros((len(column_dat), len(self.columns)))
            for clf_ind, clf_inst in enumerate(self.classifier_list):
                raw_prediction = clf_inst.predict(column_dat)
                # applying the weights to each class in the raw prediction
                for class_ind in range(len(self.columns)):
                    raw_prediction[:,class_ind] = \
                        (raw_prediction[:,class_ind] *
                         self.weights[class_ind, clf_ind])
                scores = scores + raw_prediction
            flat_scores = scores.sum(axis=0) / len(column_dat)
            max_ind = flat_scores.argmax()
            predicted_mapping[column] = self.columns[max_ind]
        return predicted_mapping
