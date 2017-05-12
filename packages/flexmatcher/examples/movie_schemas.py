import pandas as pd

import flexmatcher
# Let's assume that the mediated schema has three attributes
# movie_name, movie_year, movie_rating

# creating one sample DataFrame where the schema is (year, Movie, imdb_rating)
vals1 = [['year', 'Movie', 'imdb_rating'],
        ['2001', 'Lord of the Rings', '8.8'],
        ['2010', 'Inception', '8.7'],
        ['1999', 'The Matrix', '8.7']]
header = vals1.pop(0)
data1 = pd.DataFrame(vals1, columns=header)
# specifying mapping between schema of the dataframe and the mediated schema
data1_mapping = {'year': 'movie_year', 'imdb_rating': 'movie_rating',
                 'Movie': 'movie_name'}

# creating another sample DataFrame where the schema is
# (title, produced, popularity)
vals2 = [['title', 'produced', 'popularity'],
        ['The Godfather', '1972', '9.2'],
        ['Silver Linings Playbook', '2012', '7.8'],
        ['The Big Short', '2015', '7.8']]
header = vals2.pop(0)
data2 = pd.DataFrame(vals2, columns=header)
# specifying mapping between schema of the dataframe and the mediated schema
data2_mapping = {'popularity': 'movie_rating', 'produced': 'movie_year',
                 'title': 'movie_name'}

# creating a list of dataframes and their mappings
schema_list = [data1, data2]
mapping_list = [data1_mapping, data2_mapping]

# creating the third dataset (which is our test dataset)
# we assume that we don't know the mapping and we want FlexMatcher to find it.
vals3 = [['rt', 'id', 'yr'],
        ['8.5', 'The Pianist', '2002'],
        ['7.7', 'The Social Network', '2010']]
header = vals3.pop(0)
data3 = pd.DataFrame(vals3, columns=header)


# Using Flexmatcher
fm = flexmatcher.FlexMatcher()                       # creating an instance
fm.create_training_data(schema_list, mapping_list)   # pass the data to flexmatcher
fm.train()                                           # train flexmatcher
predicted_mapping = fm.make_prediction(data3)        # make prediction on the test data

# printing the predictions
print ('FlexMatcher predicted that "rt" should be mapped to ' +
       predicted_mapping['rt'])
print ('FlexMatcher predicted that "yr" should be mapped to ' +
       predicted_mapping['yr'])
print ('FlexMatcher predicted that "id" should be mapped to ' +
       predicted_mapping['id'])
