import pandas as pd
import unittest

import flexmatcher

class ManyColumnsTest(unittest.TestCase):

    def setUp(self):
        # creating the first dataset
        vals1 = {}
        for i in range(10):  # 10 columns
            vals1['A' + str(i)] = []
            for j in range(5):  # 5 data points
                vals1['A' + str(i)].append('hi ' + str(i * 10 + j))

        self.data1 = pd.DataFrame(vals1)
        self.data1_mapping = {}
        for i in range(10):
            self.data1_mapping['A' + str(i)] = str(i)

    def test_prediction(self):
        # Using Flexmatcher
        fm = flexmatcher.FlexMatcher()
        schema_list = [self.data1]
        mapping_list = [self.data1_mapping]
        fm.create_training_data(schema_list, mapping_list)
        fm.train()
        predicted_mapping = fm.make_prediction(self.data1)
        for i in range(10):
            self.assertEqual(predicted_mapping['A' + str(i)], str(i))
