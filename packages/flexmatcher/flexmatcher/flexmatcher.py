import pandas
import classify
from sklearn import linear_model
import numpy as np

class FlexMatcher:

    def __init__(self):
        pass

    def create_training_data(self, dataframes_list, mappings_list, sample_size=100):
        training_data_list = []
        for (dataframe, mapping) in zip(dataframes_list, mappings_list):
            # get the sample
            sampled_rows = dataframe.sample(min(sample_size, dataframe.shape[0]))
            sampled_training_data = pandas.melt(sampled_rows, value_vars=list(sampled_rows.columns))
            sampled_training_data.columns = ['name', 'value']
            sampled_training_data['class'] = sampled_training_data.apply(lambda row: mapping[row['name']], axis=1)
            training_data_list.append(sampled_training_data)
        self.training_data = pandas.concat(training_data_list)
        self.training_data.reset_index(drop=True, inplace=True)
        self.training_data = self.training_data.fillna('NA')

    def train(self):
        classifierA = classify.NaiveBayes(self.training_data)
        classifierB = classify.Tf_Idf(self.training_data)
        self.classifier_list = [classifierA, classifierB]
        self.prediction_list = [classifier.predict_training(3) for classifier in self.classifier_list]
        self.train_meta_learner()

    def train_meta_learner(self):
        (_, class_num) = self.prediction_list[0].shape
        self.class_list = sorted(list(self.training_data['class'].drop_duplicates()))
        coeff_list = []
        for class_index in range(class_num):
            class_name = self.class_list[class_index]
            # preparing the dataset for linear regression
            regression_data = self.training_data[['class']].copy()
            regression_data['is_class'] = \
                self.training_data.apply(lambda row: (row['class'] == class_name), axis=1)
            # adding the prediction probability from classifiers
            for classifier_index in range(len(self.prediction_list)):
                prediction = self.prediction_list[classifier_index]
                regression_data['classifer' + str(classifier_index)] = \
                    prediction[:,class_index]
            # setting up the linear regression
            stacker = linear_model.LinearRegression()
            stacker.fit(regression_data.iloc[:,2:], regression_data['is_class'])
            coeff_list.append(stacker.coef_.reshape(1,-1))
        self.weights = np.concatenate(tuple(coeff_list))

    def make_prediction(self, data):
        data = data.fillna('NA')
        # predicting each column
        predicted_mapping = {}
        (_, column_num) = data.shape
        for column in range(column_num):
            column_dat = data[[column]]
            column_name = column_dat.columns[0]
            column_dat.columns = ['value']
            scores = np.zeros((len(column_dat), len(self.weights)))
            for classifier_ind in range(len(self.classifier_list)):
                classifier = self.classifier_list[classifier_ind]
                raw_prediction = classifier.predict(column_dat)
                # applying the weights to each class in the raw prediction
                for class_ind in range(len(self.weights)):
                    raw_prediction[:,class_ind] = raw_prediction[:,class_ind] * self.weights[class_ind, classifier_ind]
                scores = scores + raw_prediction
            flat_scores = scores.sum(axis=0) / len(column_dat)
            max_ind = flat_scores.argmax()
            predicted_mapping[column_name] = self.class_list[max_ind]
        return predicted_mapping
