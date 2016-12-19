import flexmatcher
import pandas as pd

# The mediated schema has three attributes: movie_name, movie_year, movie_rating

# Creating the first schema, a subset of its data and the mapping to the mediated schema
vals1 = [['year', 'Movie', 'imdb_rating'],
         ['2001', 'Lord of the Rings', '8.8'],
         ['2010', 'Inception', '8.7'],
         ['1999', 'The Matrix', '8.7']]
header = vals1.pop(0)
data1 = pd.DataFrame(vals1, columns=header)
data1_mapping = {'year': 'movie_year', 'imdb_rating': 'movie_rating', 'Movie': 'movie_name'}

# Creating the second schema, a subset of its data and the mapping to the mediated schema
vals2 = [['title', 'produced', 'popularity'],
         ['The Godfather', '1972', '9.2'],
         ['Silver Linings Playbook', '2012', '7.8'],
         ['The Big Short', '2015', '7.8']]
header = vals2.pop(0)
data2 = pd.DataFrame(vals2, columns=header)
data2_mapping = {'popularity': 'movie_rating', 'produced': 'movie_year', 'title': 'movie_name'}

# Using Flexmatcher
fm = flexmatcher.FlexMatcher()
schema_list = [data1, data2]
mapping_list = [data1_mapping, data2_mapping]
fm.create_training_data(schema_list, mapping_list)
fm.train()

# Creating a test schmea
vals3 = [['rt', 'id', 'yr'],
         ['8.5', 'The Pianist', '2002'],
         ['7.7', 'The Social Network', '2010']]
header = vals3.pop(0)
data3 = pd.DataFrame(vals3, columns=header)
print fm.make_prediction(data3)
