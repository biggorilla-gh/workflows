import pandas as pd
import unittest

import flexmatcher

class MovieTest(unittest.TestCase):

    def setUp(self):
        # creating the first dataset
        vals1 = [['year', 'Movie', 'imdb_rating'],
                ['2001', 'Lord of the Rings', '8.8'],
                ['2010', 'Inception', '8.7'],
                ['1999', 'The Matrix', '8.7']]
        header = vals1.pop(0)
        self.data1 = pd.DataFrame(vals1, columns=header)
        self.data1_mapping = {'year': 'movie_year',
                              'imdb_rating': 'movie_rating',
                              'Movie': 'movie_name'}
        # creating the second dataset
        vals2 = [['title', 'produced', 'popularity'],
                ['The Godfather', '1972', '9.2'],
                ['Silver Linings Playbook', '2012', '7.8'],
                ['The Big Short', '2015', '7.8']]
        header = vals2.pop(0)
        self.data2 = pd.DataFrame(vals2, columns=header)
        self.data2_mapping = {'popularity': 'movie_rating',
                              'produced': 'movie_year',
                              'title': 'movie_name'}
        # creating the third dataset
        vals3 = [['rt', 'id', 'yr'],
                ['8.5', 'The Pianist', '2002'],
                ['7.7', 'The Social Network', '2010']]
        header = vals3.pop(0)
        self.data3 = pd.DataFrame(vals3, columns=header)
        self.data3_mapping = {'rt': 'movie_rating',
                              'yr': 'movie_year',
                              'id': 'movie_name'}

    def test_prediction(self):
        # Using Flexmatcher
        fm = flexmatcher.FlexMatcher()
        schema_list = [self.data1, self.data2]
        mapping_list = [self.data1_mapping, self.data2_mapping]
        fm.create_training_data(schema_list, mapping_list)
        fm.train()
        predicted_mapping = fm.make_prediction(self.data3)
        self.assertEqual(predicted_mapping['rt'], 'movie_rating')
        self.assertEqual(predicted_mapping['yr'], 'movie_year')
        self.assertEqual(predicted_mapping['id'], 'movie_name')
