"""
This is an example of data cleaning and analysis of a database. It operates on .csv files from the MECORE DATABASE. 
You can find the most up to date version of the database here: (https://wuegaki.ppls.ed.ac.uk/mecore/mecore-databases/)
Example .csv files are available with this file in the folder csv_files_070823

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
import json

import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words

from sklearn.tree import DecisionTreeClassifier

from data_cleaning import DataCleaner, AutomaticCleaner, HypothesesFinder, DataExtractor, ValuesExtractor

#Import database to a pandas database. 
df_1 = pd.read_csv('csv_files_070823/PolishTablePol001V0.csv')

#Create the values_of_columns.json you can edit the file to tune the cleaner to your needs.
ext = ValuesExtractor()
vals = ext.values_from_all_columns(df_1)
with open("values_of_columns.json", "w") as file:
    json.dump(vals, file)

# #Clean the database WARNING: Cleaning may take some time to run, as it checks each cell for typos. 
# cleaner = AutomaticCleaner()
# df_1 = cleaner.clean(df_1)

# #Save the cleaned database
# df_1.to_csv("cleaned_df.csv", index=False)

# df_2 = df_1
# df_3 = df_1

# #Compare datasbes predicate by predicate:

# cleaner.compare_databases(df_1, df_2)

# # Merge databases 
# df_bin = cleaner.binary_merge(df_1,df_2)
# df_merged = cleaner.merge(df_1,df_2,df_3) 


# #Save the merged database
# df_merged.to_csv("merged_df.csv", index=False)

## Tree-based hypotheses discovery
#Import a cleanded database to a pandas database. 
#df = pd.read_csv('cleaned_df.csv')

##Clean the database WARNING: Cleaning may take some time to run, as it checks each cell for typos. 

##Extracts semantic properties of the database, which are defined for anti-rogative and responsive predicates and 
# extractor = DataExtractor()
# df= extractor.get_anti_rogative_vs_responsive_df(df)

## Split into train and label
# X = df.drop(['predicate', 'label'], axis=1)
# y=df.label

# #Turn into binary values for each column
# X = pd.get_dummies(X)

# #Fit a decision tree classifier
# model = DecisionTreeClassifier()
# model.fit(X,y)

##Get hypoteses from the model
#finder = HypothesesFinder()
# hypotheses = finder.retreive_text_branches(model, X)

# #Remove hypotheses that explain less than LIMIT_OF_SAMPLES (default=3) predicates and print the hypotheses:
# hypotheses = finder.remove_overfitting(hypotheses)
# for hypothesis in hypotheses:
#     print(hypothesis)

## Discover new hypotheses using the forest based alghoritm:
# hypotheses = finder.forest_based_discovery(X,y, limit=2)
# print(hypotheses)
# print(len(hypotheses))

# #Investigate whether a set of predicates X (rows) satisfy a simple rule on label y defined by one of the properties (e.g. if all neg-raising predicates are anti-rogative):
# # It not only prints the answer but also returns a list of dictionaries with hypothesis, label and indexes of predicates from the original database that satisfy it.
# insight = finder.individual_check(X,y)
# print(insight)
# #If you want to investigate a subset of properties available in X use:
# finder.individual_check(X,y,['Preference_incompatible', 'Complement projection/reversion through negation_neither', 'neg-raising_0'])

# #Investigate a conjunctive hypothesis - tests all the combinations of values for specified list of properties conjunctions of all of them (if subsets=True) it also checks all the subsets.
# # It not only prints the answer but also returns a list of dictionaries with hypothesis, label and indexes of predicates from the original database that satisfy it.
#insight = finder.conjunctive_check(X,y, ['Preference_always', 'veridicality/anti-veridicality wrt declaratives_neither'])
# insight = finder.conjunctive_check(X,y,['Preference_incompatible', 'Complement projection/reversion through negation_neither', 'neg-raising_1'],subsets=True)
#print(insight)



## Forest-based hypotheses discovery

##To follow the cleaning steps manually create an instance of a DataCleaner
# cleaner = DataCleaner()

# # Delete empty rows and columns and check whether the columns are named correctly: look at the prompt.
# df = cleaner.drop_empty_rows(df)
# df = cleaner.drop_empty_columns(df)
# cleaner.check_unnamed(df)
# cleaner.check_names(df)

# # Remove annotation and typos. Bring all words to lowercase.  
# df = cleaner.replace_nonalphanumeric(df)
# df= cleaner.automatically_remove_typos(df)

## To accept every change made by spell checker use:
#df= cleaner.manually_remove_typos(df)

## Unify the values in each column with the values specified in values_of_columns.json
# cleaner.fix_column_values(df)

# # You can check if the values were replaced correctly using:
# values = cleaner.values_from_all_columns(df)

