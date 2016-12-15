
# coding: utf-8

# # Part 1: Data Acquistion
# --------------------------
# BigGorilla recommends a list of tools for different data acquisition tasks (See [here]()). Among these tools, **urllib** is a popular python package for fetching data across the web. In this part, we use **urllib** to download the datasets that we need for this tutorial.
# 
# ### Step 1: downloading the "Kaggle 5000 Movie Dataset"
# The desired dataset is a _.csv_ file with a url that is specified in the code snippet below.

# In[1]:

# Importing urlib (BigGorilla's recommendation for data acquisition from the web)
import urllib
import os

# Creating the data folder
if not os.path.exists('./data'):
    os.makedirs('./data')

# Obtaining the dataset using the url that hosts it
kaggle_url = 'https://github.com/sundeepblue/movie_rating_prediction/raw/master/movie_metadata.csv'
if not os.path.exists('./data/kaggle_dataset.csv'):     # avoid downloading if the file exists
    response = urllib.urlretrieve(kaggle_url, './data/kaggle_dataset.csv')


# ### Step 2: downloading the "IMDB Plain Text Data"
# The IMDB Plain Text Data (see [here](ftp://ftp.funet.fi/pub/mirrors/ftp.imdb.com/pub/)) is a collection of files where each files describe one or a few attributes of a movie. We are going to focus on a subset of movie attribues which subsequently means that we are only interested in a few of these files which are listed below:
# 
# * genres.list.gz
# * ratings.list.gz
# 
# _** Note: The total size of files mentioned above is roughly 30M. Running the following code may take a few minutes._

# In[2]:

import gzip

# Obtaining IMDB's text files
imdb_url_prefix = 'ftp://ftp.funet.fi/pub/mirrors/ftp.imdb.com/pub/'
imdb_files_list = ['genres.list.gz', 'ratings.list.gz']
for name in imdb_files_list:
    if not os.path.exists('./data/' + name):
        response = urllib.urlretrieve(imdb_url_prefix + name, './data/' + name)
        urllib.urlcleanup()   # urllib fails to download two files from a ftp source. This fixes the bug!
        with gzip.open('./data/' + name) as comp_file, open('./data/' + name[:-3], 'w') as reg_file:
            file_content = comp_file.read()
            reg_file.write(file_content)


# ### Step 3: downloading the "IMDB Prepared Data"
# During this tutorial, we discuss how the contents of _genres.list.gz_ and _ratings.list.gz_ files can be integrated. However, to make the tutorial more concise, we avoid including the same process for all the files in the "IMDB Plain Text Data". The "IMDB Prepared Data" is the dataset that we obtained by integrating a number of files from the "IMDB Plain Text Data" which we will use during later stages of this tutorial. The following code snippet downloads this dataset.

# In[3]:

imdb_url = 'https://anaconda.org/BigGorilla/datasets/1/download/imdb_dataset.csv'
if not os.path.exists('./data/imdb_dataset.csv'):     # avoid downloading if the file exists
    response = urllib.urlretrieve(kaggle_url, './data/imdb_dataset.csv')


# -----

# # Part 2: Data Extraction
# -----------------
# The "Kaggle 5000 Movie Dataset" is stored in a _.csv_ file which is alreday structured and ready to use. On the other hand, the "IMDB Plain Text Data" is a collection of semi-structured text files that need to be processed to extract the data. A quick look at the first few lines of each files shows that each file has a different format and has to be handled separately.
# 
# ##### Content of "ratings.list" data file

# In[4]:

with open("./data/ratings.list") as myfile:
    head = [next(myfile) for x in range(38)]
print (''.join(head[28:38]))   # skipping the first 28 lines as they are descriptive headers


# ##### Content of the "genres.list" data file

# In[5]:

with open("./data/genres.list") as myfile:
    head = [next(myfile) for x in range(392)]
print (''.join(head[382:392]))   # skipping the first 382 lines as they are descriptive header


# ### Step 1: Extracting the information from "genres.list"
# The goal of this step is to extract the movie titles and their production year from "movies.list", and store the extracted data into a dataframe. Dataframe (from the python package **pandas**) is one of the key BigGorilla's recommendation for data profiling and cleaning. To extract the desired information from the text, we rely on **regular expressions** which are implemented in the python package "**re**".

# In[6]:

import re
import pandas as pd

with open("./data/genres.list") as genres_file:
    raw_content = genres_file.readlines()
    genres_list = []
    content = raw_content[382:]
    for line in content:
        m = re.match(r'"?(.*[^"])"? \(((?:\d|\?){4})(?:/\w*)?\).*\s((?:\w|-)+)', line.strip())
        genres_list.append([m.group(1), m.group(2), m.group(3)])
    genres_data = pd.DataFrame(genres_list, columns=['movie', 'year', 'genre'])


# ### Step 2: Extracting the information from "ratings.list"

# In[7]:

with open("./data/ratings.list") as ratings_file:
    raw_content = ratings_file.readlines()
    ratings_list = []
    content = raw_content[28:]
    for line in content:
        m = re.match(r'(?:\d|\.|\*){10}\s+\d+\s+(1?\d\.\d)\s"?(.*[^"])"? \(((?:\d|\?){4})(?:/\w*)?\)', line.strip())
        if m is None: continue
        ratings_list.append([m.group(2), m.group(3), m.group(1)])
    ratings_data = pd.DataFrame(ratings_list, columns=['movie', 'year', 'rating'])


# Note that one has to repeat the information extraction procedure for other data files as well if he is interested in their content. For now (and to keep the tutorial simple), we assume that we are only interested in genres and ratings of movies. The above code snippets store the extracted data on these two attributes into two dataframes (namely, **genres_list** and **ratings_list**).
# 
# ------

# # Part 3: Data Profiling & Cleaning
# ---------------------------
# 
# The high-level goal in this stage of data prepration is to look into the data that we have acquired and extracted so far. This helps us to get familiar with data, understand in what ways the data needs cleaning or transformation, and finally enables us to prepare the data for the following steps of the data integration task.
# 
# ### Step 1: Loading the "Kaggle 5000 Movies Dataset"
# 
# According to BigGorilla, dataframes (from the python package **pandas**) are suitable for data exploration and data profiling. In [Part 2](https://github.com/rit-git/BigGorilla/blob/tutorial/Tutorial/Part%202%20--%20Data%20Extraction.ipynb) of the tutorial, we stored the extracted data from "IMDB Plain Text Data" into dataframes. It would be appropriate to load the "Kaggle 5000 Movies Dataset" into a dataframe as well and follow the same data profiling procedure for all datasets.

# In[8]:

import pandas as pd

# Loading the Kaggle dataset from the .csv file (kaggle_dataset.csv)
kaggle_data = pd.read_csv('./data/kaggle_dataset.csv')


# ### Step 2: Calculating some basic statistics (profiling)
# 
# Let's start by finding out how many movies are listed in each dataframe.

# In[9]:

print ('Number of movies in kaggle_data: {}'.format(kaggle_data.shape[0]))
print ('Number of movies in genres_data: {}'.format(genres_data.shape[0]))
print ('Number of movies in ratings_data: {}'.format(ratings_data.shape[0]))


# We can also check to see if we have duplicates (i.e., a movie appearing more than once) in the data. We consider an entry duplicate if we can find another entry with the same movie title and production year.

# In[10]:

print ('Number of duplicates in kaggle_data: {}'.format(
    sum(kaggle_data.duplicated(subset=['movie_title', 'title_year'], keep=False))))
print ('Number of duplicates in genres_data: {}'.format(
    sum(genres_data.duplicated(subset=['movie', 'year'], keep=False))))
print ('Number of duplicates in ratings_data: {}'.format(
    sum(ratings_data.duplicated(subset=['movie', 'year'], keep=False))))


# ### Step 3: Dealing with duplicates (cleaning)
# 
# There are many strategies to deal with duplicates. Here, we are going to use a simple method for dealing with duplicates and that is to only keep the first occurrence of a duplicated entry and remove the rest.

# In[11]:

kaggle_data = kaggle_data.drop_duplicates(subset=['movie_title', 'title_year'], keep='first').copy()
genres_data = genres_data.drop_duplicates(subset=['movie', 'year'], keep='first').copy()
ratings_data = ratings_data.drop_duplicates(subset=['movie', 'year'], keep='first').copy()


# ### Step 4: Normalizing the text (cleaning)
# 
# The key attribute that we will use to integrate our movie datasets is the movie titles. So it is important to normalize these titles. The following code snippet makes all movie titles lower case, and then removes certain characters such as "'" and "?", and replaces some other special characters (e.g., "&" is replaced with "and"). 

# In[12]:

def preprocess_title(title):
    title = title.lower()
    title = title.replace(',', ' ')
    title = title.replace("'", '')    
    title = title.replace('&', 'and')
    title = title.replace('?', '')
    title = title.decode('utf-8', 'ignore')
    return title.strip()

kaggle_data['norm_movie_title'] = kaggle_data['movie_title'].map(preprocess_title)
genres_data['norm_movie'] = genres_data['movie'].map(preprocess_title)
ratings_data['norm_movie'] = ratings_data['movie'].map(preprocess_title)


# ### Step 5: Looking at a few samples
# 
# The goal here is to a look at a few sample entries from each dataset for a quick sanity check. To keep the tutorial consice, we just present this step for the "Kaggle 5000 Movies Dataset" which is stored in the **kaggle_data** dataframe. 

# In[13]:

kaggle_data.sample(3, random_state=0)


# Looking at the data guides us to decide in what ways we might want to clean the data. For instance, the small sample data shown above, reveals that the **title_year** attribute is stored as floats (i.e., rational numbers). We can add another cleaning step to transform the **title_year** into strings and replace the missing title years with symbol **"?"**.

# In[14]:

def preprocess_year(year):
    if pd.isnull(year):
        return '?'
    else:
        return str(int(year))

kaggle_data['norm_title_year'] = kaggle_data['title_year'].map(preprocess_year)
kaggle_data.head()


# -----

# # Part 4: Data Matching & Merging
# -------------------------
# The main goal in this part is go match the data that we have acquired from different sources to create a single rich dataset. Recall that in [Part 3](https://github.com/rit-git/BigGorilla/blob/tutorial/Tutorial/Part%203%20--%20Data%20Profiling%20%26%20Cleaning.ipynb), we transformed all datasets into a dataframe which we used to clean the data. In this part, we continue using the same dataframes for the data that we have prepared so far.
# 
# ### Step 1: Integrating the "IMDB Plain Text Data" files
# Note that both **ratings_data** and **genres_data** dataframes contain data that come from the same source (i.e., "the IMDB Plain Text data"). Thus, we assume that there are no inconsistencies between the data stored in these dataframe and to combine them, all we need to do is to match the entries that share the same title and production year. This simple "exact match" can be done simply using dataframes.

# In[15]:

brief_imdb_data = pd.merge(ratings_data, genres_data, how='inner', on=['norm_movie', 'year'])
brief_imdb_data.head()


# We refer to the dataset created above as the **brief_imdb_data** since it only contains two attributes (namely, genre and rating). Henceforth, we are going to use a richer version of the IMDB dataset which we created by integrating a number of files from the "IMDB Plain Text Data". If you have completed the first part of this tutorial, then this dataset is already downloaded and stored in *"imdb_dataset.csv"* under the _"data"_ folder. The following code snippet loads this dataset, does preprocessing on the title and production year of movies, removes the duplicates as before, and prints the size of the dataset.

# In[16]:

# reading the new IMDB dataset
imdb_data = pd.read_csv('./data/imdb_dataset.csv')
# let's normlize the title as we did in Part 3 of the tutorial
imdb_data['norm_title'] = imdb_data['title'].map(preprocess_title)
imdb_data['norm_year'] = imdb_data['year'].map(preprocess_year)
imdb_data = imdb_data.drop_duplicates(subset=['norm_title', 'norm_year'], keep='first').copy()
imdb_data.shape


# ### Step 2: Integrating the Kaggle and IMDB datasets
# 
# A simple approach to integrate the two datasets is to simply join entries that share the same movie title and year of production. The following code reveals that 4,248 matches are found using this simple approach.

# In[17]:

data_attempt1 = pd.merge(imdb_data, kaggle_data, how='inner', left_on=['norm_title', 'norm_year'],
                         right_on=['norm_movie_title', 'norm_title_year'])
data_attempt1.shape


# But given that IMDB and Kaggle datasets are collected from different sources, chances are that the name of a movie would be slightly different in these datasets (e.g. "Wall.E" vs "WallE"). To be able to find such matches, one can look at the similarity of movie titles and consider title with high similarity to be the same entity. BigGorilla's recommendation for doing similarity join across two datasets is the python package **py_stringsimjoin**. The following code snippet uses the **py_stringsimjoin** to match all the titles that have an edit distance of one or less (i.e., there is at most one character that needs to be changed/added/removed to make both titles identical). Once the similarity join is complete, it only selects the title pairs that are produced in the same year.

# In[18]:

import py_stringsimjoin as ssj
import py_stringmatching as sm

imdb_data['id'] = range(imdb_data.shape[0])
kaggle_data['id'] = range(kaggle_data.shape[0])
similar_titles = ssj.edit_distance_join(imdb_data, kaggle_data, 'id', 'id', 'norm_title',
                                        'norm_movie_title', l_out_attrs=['norm_title', 'norm_year'],
                                         r_out_attrs=['norm_movie_title', 'norm_title_year'], threshold=1)
# selecting the entries that have the same production year
data_attempt2 = similar_titles[similar_titles.r_norm_title_year == similar_titles.l_norm_year]
data_attempt2.shape


# We can see that using the similarity join 4,689 titles were matched. Let's look at some of the titles that are matched by the similarity join but are not identical.

# In[19]:

data_attempt2[data_attempt2.l_norm_title != data_attempt2.r_norm_movie_title].head()


# While instances such as "walle" and "wall.e" are correctly matched, we can see that this techniques also makes some errors (e.g., "grave" and "brave"). This raises the following questions: "what method should be used for data matching?" and "how can we determine the quality of the matching?". BigGorilla's recommendation for dealing with this problem is using the pythong package **py_entitymatching** which is developed as part of the [Magellan project](https://sites.google.com/site/anhaidgroup/projects/magellan).
# 
# In the next step, we demonstrate how **py_entitymatching** uses machine learning techniques for the data-matching purposes as well as how it enables us to evaluate the quality of the produced matching.
# 
# ### Step 3: Using Magellan for data matching
# 
# #### Substep A: Finding a candiate set (Blocking)
# The goal of this step is to limit the number of pairs that we consider as potential matches using a simple heuristic. For this task, we can create a new column in each dataset that combines the values of important attributes into a single string (which we call the **mixture**). Then, we can use the string similarity join as before to find a set of entities that have some overlap in the values of the important columns. Before doing that, we need to transform the columns that are part of the mixture to strings. The **py_stringsimjoin** package allows us to do so easily.

# In[20]:

# transforming the "budget" column into string and creating a new **mixture** column
ssj.utils.converter.dataframe_column_to_str(imdb_data, 'budget', inplace=True)
imdb_data['mixture'] = imdb_data['norm_title'] + ' ' + imdb_data['norm_year'] + ' ' + imdb_data['budget']

# repeating the same thing for the Kaggle dataset
ssj.utils.converter.dataframe_column_to_str(kaggle_data, 'budget', inplace=True)
kaggle_data['mixture'] = kaggle_data['norm_movie_title'] + ' ' + kaggle_data['norm_title_year'] +                          ' ' + kaggle_data['budget']


# Now, we can use the **mixture** columns to create a desired candiate set which we call **C**.

# In[21]:

C = ssj.overlap_coefficient_join(kaggle_data, imdb_data, 'id', 'id', 'mixture', 'mixture', sm.WhitespaceTokenizer(), 
                                 l_out_attrs=['norm_movie_title', 'norm_title_year', 'duration',
                                              'budget', 'content_rating'],
                                 r_out_attrs=['norm_title', 'norm_year', 'length', 'budget', 'mpaa'],
                                 threshold=0.65)
C.shape


# We can see that by doing a similarity join, we already reduced the candidate set to 18,317 pairs.
# 
# #### Substep B: Specifying the keys 
# The next step is to specify to the **py_entitymatching** package which columns correspond to the keys in each dataframe. Also, we need to specify which columns correspond to the foreign keys of the the two dataframes in the candidate set.

# In[22]:

import py_entitymatching as em
em.set_key(kaggle_data, 'id')   # specifying the key column in the kaggle dataset
em.set_key(imdb_data, 'id')     # specifying the key column in the imdb dataset
em.set_key(C, '_id')            # specifying the key in the candidate set
em.set_ltable(C, kaggle_data)   # specifying the left table 
em.set_rtable(C, imdb_data)     # specifying the right table
em.set_fk_rtable(C, 'r_id')     # specifying the column that matches the key in the right table 
em.set_fk_ltable(C, 'l_id')     # specifying the column that matches the key in the left table 


# 
# #### Subset C: Debugging the blocker
# 
# Now, we need to make sure that the candidate set is loose enough to include pairs of movies that are not very close. If this is not the case, there is a chance that we have eliminated pair that could be potentially matched together. By looking at a few pairs from the candidate set, we can judge whether the blocking step has been too harsh or not.
# 
# *Note: The **py_entitymatching** package provides some tools for debugging the blocker as well.*

# In[23]:

C[['l_norm_movie_title', 'r_norm_title', 'l_norm_title_year', 'r_norm_year',
   'l_budget', 'r_budget', 'l_content_rating', 'r_mpaa']].head()


# Based on the above sample we can see that the blocking seems to be reasonable.
# 
# #### Substep D: Sampling from the candiate set
# 
# The goal of this step is to obtain a sample from the candidate set and manually label the sampled candidates; that is, to specify if the candiate pair is a correct match or not.

# In[24]:

# Sampling 500 pairs and writing this sample into a .csv file
sampled = C.sample(500, random_state=0)
sampled.to_csv('./data/sampled.csv', encoding='utf-8')


# In order to label the sampled data, we can create a new column in the _.csv_ file (which we call **label**) and put value 1 under that column if the pair is a correct match and 0 otherwise. To avoid overriding the files, let's rename the new file as **labeled.csv**.

# In[25]:

# If you would like to avoid labeling the pairs for now, you can download the labled.csv file from
# BigGorilla using the following command (if you prefer to do it yourself, commend the next line)
response = urllib.urlretrieve('https://anaconda.org/BigGorilla/datasets/1/download/labeled.csv',
                              './data/labeled.csv')
labeled = em.read_csv_metadata('data/labeled.csv', ltable=kaggle_data, rtable=imdb_data,
                               fk_ltable='l_id', fk_rtable='r_id', key='_id')
labeled.head()


# #### Substep E: Traning machine learning algorithms
# 
# Now we can use the sampled dataset to train various machine learning algorithms for our prediction task. To do so, we need to split our dataset into a training and a test set, and then select the desired machine learning techniques for our prediction task.

# In[26]:

split = em.split_train_test(labeled, train_proportion=0.5, random_state=0)
train_data = split['train']
test_data = split['test']

dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
nb = em.NBMatcher(name='NaiveBayes')


# Before we can apply any machine learning technique, we need to extract a set of features. Fortunately, the **py_entitymatching** package can automatically extract a set of features once we specify which columns in the two datasets correspond to each other. The following code snippet starts by specifying the correspondence between the column of the two datasets. Then, it uses the **py_entitymatching** package to determine the type of each column. By considering the types of columns in each dataset (stored in variables *l_attr_types* and *r_attr_types*), and using the tokenizers and similarity functions suggested by the package, we can extract a set of instructions for extracting features. Note that variable **F** is not the set of extracted features, rather it encodes the instructions for computing the features.

# In[27]:

attr_corres = em.get_attr_corres(kaggle_data, imdb_data)
attr_corres['corres'] = [('norm_movie_title', 'norm_title'), 
                         ('norm_title_year', 'norm_year'),
                        ('content_rating', 'mpaa'),
                         ('budget', 'budget'),
]

l_attr_types = em.get_attr_types(kaggle_data)
r_attr_types = em.get_attr_types(imdb_data)

tok = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()

F = em.get_features(kaggle_data, imdb_data, l_attr_types, r_attr_types, attr_corres, tok, sim)


# Given the set of desired features **F**, we can now calculate the feature values for our training data and also impute the missing values in our data. In this case, we choose to replace the missing values with the mean of the column.

# In[28]:

train_features = em.extract_feature_vecs(train_data, feature_table=F, attrs_after='label', show_progress=False) 
train_features = em.impute_table(train_features,  exclude_attrs=['_id', 'l_id', 'r_id', 'label'], strategy='mean')


# Using the calculated features, we can evaluate the performance of different machine learning algorithms and select the best one for our matching task.

# In[29]:

result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=train_features, 
                           exclude_attrs=['_id', 'l_id', 'r_id', 'label'], k=5,
                           target_attr='label', metric='f1', random_state=0)
result['cv_stats']


# We can observe based on the reported accuracy of different techniques that the "random forest (RF)" algorithm achieves the best performance. Thus, it is best to use this technique for the matching.

# #### Substep F: Evaluating the quality of our matching
# 
# It is important to evaluate the quality of our matching. We can now, use the traning set for this purpose and measure how well the random forest predicts the matches. We can see that we are obtaining a high accuracy and recall on the test set as well.

# In[30]:

best_model = result['selected_matcher']
best_model.fit(table=train_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], target_attr='label')

test_features = em.extract_feature_vecs(test_data, feature_table=F, attrs_after='label', show_progress=False)
test_features = em.impute_table(test_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], strategy='mean')

# Predict on the test data
predictions = best_model.predict(table=test_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], 
                                 append=True, target_attr='predicted', inplace=False)

# Evaluate the predictions
eval_result = em.eval_matches(predictions, 'label', 'predicted')
em.print_eval_summary(eval_result)


# #### Substep G: Using the trained model to match the datasets
# 
# Now, we can use the trained model to match the two tables as follows:

# In[31]:

candset_features = em.extract_feature_vecs(C, feature_table=F, show_progress=True)
candset_features = em.impute_table(candset_features, exclude_attrs=['_id', 'l_id', 'r_id'], strategy='mean')
predictions = best_model.predict(table=candset_features, exclude_attrs=['_id', 'l_id', 'r_id'],
                                 append=True, target_attr='predicted', inplace=False)
matches = predictions[predictions.predicted == 1] 


# Note that the **matches** dataframe contains many columns storing the extracted features for both datasets. The following code snippet removes all the unnecessary columns and creates a nice formatted dataframe that has the resulting integrated dataset.

# In[32]:

from py_entitymatching.catalog import catalog_manager as cm
matches = matches[['_id', 'l_id', 'r_id', 'predicted']]
matches.reset_index(drop=True, inplace=True)
cm.set_candset_properties(matches, '_id', 'l_id', 'r_id', kaggle_data, imdb_data)
matches = em.add_output_attributes(matches, l_output_attrs=['norm_movie_title', 'norm_title_year', 'budget', 'content_rating'],
                                   r_output_attrs=['norm_title', 'norm_year', 'budget', 'mpaa'],
                                   l_output_prefix='l_', r_output_prefix='r_',
                                   delete_from_catalog=False)
matches.drop('predicted', axis=1, inplace=True)
matches.head()

