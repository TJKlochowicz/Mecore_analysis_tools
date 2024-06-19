""" 
You can use the AutomaticCleaner and DataCleaner class to clean and merge databases.
To clean automatically use function .clean of the AutomaticCleaner
To clean step by step use the DataCleaner
After cleaning each database you can merge databases using the merging functions. 

You can analyse databases using the HypothesesFinder class. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import itertools

import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words

from sklearn.tree import DecisionTreeClassifier


CORRECT_DESCRIPTIVE_COLUMN_NAMES = ['language', 'predicate', 'English translation', 'predicate class'] #Default: [predicate, english translation, predicate class]
SPECIFIC_WORDS = ['implies', 'anti-veridical', 'neg-raising'] #Add non-standard/domain specific words to treat them as non-typos.
PROBLEMATIC_WORDS = ['implex', 'implead', 'implete'] #Remove words that cause incorrect behaviour.
DESCRIPTIVE_COLUMNS = len(CORRECT_DESCRIPTIVE_COLUMN_NAMES) #How many first columns are descriptive.

LIMIT_OF_SAMPLES = 2 # Hypotheses, which explain this or lower number of samples are not output by the alghoritm. 
MAXIMAL_NUMBER_OF_PROPERTIES = 4 #Maximal number of properties, which can constiture a hypothesis. 

class ValuesExtractor:
    def __init__(self):
        self.nan_values = ["N/A", "nan", "NaN", "Not applicable", "not applicable"]
    def values_from_column(self, column):
        values = column.value_counts().index.tolist()
        return values

    def values_from_all_columns(self, df):
        values = {}
        for column in df.columns[DESCRIPTIVE_COLUMNS:]:
            values[column] = self.values_from_column(df[column])
        return values

class DataCleaner:
    def __init__(self):
        self.nan_values = ["N/A", "nan", "NaN", "Not applicable", "not applicable"]
        self.correct_words = words.words() 
        with open ('values_of_columns.json', 'r') as file:
            self.values_of_columns = json.load(file)
        with open ('semantic_values_of_columns.json', 'r') as file:
            self.semantic_values_of_columns = json.load(file)
        self.correct_column_names = []
        self.refine_dictionnary()
        self.define_column_names()
    
    def refine_dictionnary(self):
        self.correct_words.extend(SPECIFIC_WORDS)
        for word in PROBLEMATIC_WORDS:
            self.correct_words.remove(word)
        
    
    def define_column_names(self):

        for item in CORRECT_DESCRIPTIVE_COLUMN_NAMES:
            self.correct_column_names.append(item)

        for key in self.values_of_columns:
            self.correct_column_names.append(key)
        


# Commands to drop empty rows and columns

    def drop_empty_columns(self,df):
        clean = df.dropna(how='all',axis=1)
        return clean
        
    def drop_empty_rows(self, df):
        clean = df.dropna(how='all')
        return clean

    def drop_empty_all(self,df):
        clean = self.drop_empty_columns(df)
        clean = self.drop_empty_rows(clean)
        return clean
    
# Commands to check if columns were renamed properly
    def check_unnamed(self, df):
        for name in df.columns:
            if "Unnamed" in name.split(":"): 
                print("Warning: file contains unnamed columns")
                break

    def drop_undefined_columns(self, df):
        for column in df.columns.tolist():
            if column not in self.correct_column_names:
                df = df.drop(column, axis=1)
        return df
    
    def inted(self, value):
        try:
            int(value)
            return int(value)
        except:
            return None

    def check_names(self,df, semantic_only=False):
        fixed = True
        error = 0
        if semantic_only:
            correct_names = CORRECT_DESCRIPTIVE_COLUMN_NAMES + list(self.semantic_values_of_columns.keys())
        else:
            correct_names = self.correct_column_names
        print(f"The databse has {len(df.columns)} columns and {len(df.index)} rows.")
        listed_column_names = df.columns.tolist()
        # print(correct_names)
        # print(listed_column_names)
        if listed_column_names == correct_names:
            print("Columns are named correctly!")
            return 0
        else:
            print("The following names do not match with the standard naming:")
            for i in range(len(listed_column_names)):
                try:
                    if len(listed_column_names) > len(self.correct_column_names):
                        for _ in range(len(listed_column_names) - len(self.correct_column_names) + 1):
                            self.correct_column_names.append('Dummy column')
                    if listed_column_names[i] != self.correct_column_names[i]:
                        print("Name in your database: " + listed_column_names[i] + " Correct name: " + self.correct_column_names[i])
                except IndexError:
                    print("There are too few columns!")
                    error = 1
                    break
            fix = input("Do you want to change the names to the correct ones? (y/n): ")
            if fix.lower().strip()[0] == 'y' and error == 0:
                for i in range(len(listed_column_names)):
                    if listed_column_names[i] != self.correct_column_names[i]:
                        drop = input(f"Would you like to drop {listed_column_names[i]}? (y/n): ")
                        if drop.lower().strip()[0] == 'y':
                            df.drop(f'{str(listed_column_names[i])}', axis=1, inplace=True)
                            print(f"Column {listed_column_names[i]} was dropped. Restarting the procedure!")
                            self.check_names(df)
                        agree = input(f"Would you like to replace {listed_column_names[i]} with {self.correct_column_names[i]}? (y/n): ")
                        if agree.lower().strip()[0] == 'y':
                            df.rename(columns = {listed_column_names[i]: self.correct_column_names[i]}, inplace = True)
                        else: 
                            print(f'Name {listed_column_names[i]} was not fixed!')
                            fixed = False
            elif error == 1:
                print('Names were not fixed, because there are too few columns!')
                fixed = False
                return 1
            else:
                print('Names were not fixed!')
                fixed = False
                return 1
            self.define_column_names()
            if fixed:
                return 0
            else:
                return 2
    
    # Remove nonalphanumeric characters like '*' from the cells. Good to run before evoving typos. 
    def replace_nonalphanumeric(self, df):
        for value in self.nan_values:
            df = df.replace(value, np.nan)
        for i in range(df.shape[0]): #iterate over rows
            for j in range(df.shape[1]): #iterate over columns
                if pd.isna(df.iat[i,j]):
                    pass
                elif type(df.iat[i,j]) is float:
                    df.iat[i,j] = int(df.iat[i,j])
                else:
                    df.iat[i,j] = re.sub('[^0-9a-zA-Z]+ ', '', str(df.iloc[i][j]))
                    df.iat[i,j] = re.sub('\*', '', str(df.iloc[i][j]))
                    for _ in range(9):
                        df.iat[i,j] = re.sub('  ', ' ', str(df.iloc[i][j]))
        for value in self.nan_values:
            df = df.replace(value, np.nan)
        return df
    
# Typos removers

    # Remove all the typos from the cells (based on a spellchecker, so can make mistakes. Run check_columns_values afterwards)
    def automatically_remove_typos(self, df):  
        for i in range(df.shape[0]): #iterate over rows
            for j in range(DESCRIPTIVE_COLUMNS,df.shape[1]): #iterate over columns
                if type(df.iat[i,j]) is str and len(df.iat[i,j]) > 0 and not df.iat[i,j].isdigit():
                    new_words = []   
                    for word in df.iat[i, j].split(' '):
                        if len(word) > 1 and type(word) == str and not word.isdigit():
                            word = word.lower()
                            temp = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))),w) for w in self.correct_words if w[0]==word[0]]
                            if len(temp) > 0:
                                new_words.append(sorted(temp, key = lambda val:val[0])[0][1])
                    df.iat[i,j] = ' '.join(new_words)
        return df
    
    # Remove typos cell by cell. Each time it finds a typo, the user is prompted to accept the replacement, or provide their own. (takes a lot of time not recommended)
    def manually_remove_typos(self,df):
        for i in range(df.shape[0]): #iterate over rows
            for j in range(DESCRIPTIVE_COLUMNS,df.shape[1]): #iterate over columns
                if type(df.iat[i,j]) is str and len(df.iat[i,j]) > 0 and not df.iat[i,j].isdigit():
                    new_words = []   
                    for word in df.iat[i, j].split(' '):
                        if len(word) > 1 and type(word) == str and not word.isdigit():
                            word = word.lower()
                            temp = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))),w) for w in self.correct_words if w[0]==word[0]]
                            if len(temp) > 0:
                                if sorted(temp, key = lambda val:val[0])[0][1] == word:
                                    new_words.append(sorted(temp, key = lambda val:val[0])[0][1])
                                else:
                                    agreed = input(f"Do you want to replace '{word}' with '{sorted(temp, key = lambda val:val[0])[0][1]}' in column '{df.columns[j]}'? (y/n): ")
                                    if agreed.lower().strip()[0] == 'y':
                                        new_words.append(sorted(temp, key = lambda val:val[0])[0][1])
                                    elif agreed.lower().strip()[0] == "n":
                                            replace = input(f"Do you want to replace '{word}' with another word/phrase? (y/n): ")
                                            if replace.lower().strip()[0] == 'y':
                                                new_word = input("Type in the new word/phrase: ")
                                                new_words.append(new_word)
                                                print(f"Replaced '{word}' with '{new_word}'!")
                                            elif replace.lower().strip()[0] == "n":
                                                print(f"'{word}' was not replaced!")
                                                new_words.append(word)
                                            else:
                                                print("You can only answer 'y' or 'n'!")
                                                new_words.append(word)
                                                print(f"'{word}' was not replaced!")
                                    else:
                                        print("You can only answer 'y' or 'n'!")
                                        new_words.append(word)
                                        print(f"'{word}' was not replaced!")

                    df.iat[i,j] = ' '.join(new_words)
        return df
            
# Cell values unification commands

    def values_from_column(self, column):
        values = column.value_counts().index.tolist()
        return values

    def values_from_all_columns(self, df):
        values = {}
        for column in df.columns[DESCRIPTIVE_COLUMNS:]:
            values[column] = self.values_from_column(df[column])
        return values
    
    def fix_column_values(self, df):
        for column in df.columns[DESCRIPTIVE_COLUMNS:]:
            column_index = df.columns.get_loc(column)
            for value in df[column].value_counts().index.tolist():
                if value not in self.values_of_columns[column] and self.inted(value) not in self.values_of_columns[column] and str(self.inted(value)) not in self.values_of_columns[column]:
                    if pd.isna(value) or value =='':
                        print(f"There are {df[column].value_counts()[value]} empty cells in the column '{column}'")
                    else:
                        print(f"There are {df[column].value_counts()[value]} cells with value '{value}' in the column '{column}', which is not specified for this column.")
                        incorrect_input = True
                        while incorrect_input:
                            agreed = input("Would you like to replace all these values with a correct one? (y/n): ")
                            if agreed.lower().strip()[0] == 'y':
                                new_value = input(f"The specified values for this column are:{self.values_of_columns[column]} \n \n " + "Provide the correct value: ").strip().lower()
                                if new_value in self.values_of_columns[column]:
                                    for i in range(df.shape[0]):
                                        if df.iat[i,column_index] == value:
                                            df.iat[i,column_index] = new_value
                                    print(f"All values '{value}' replaced with '{new_value}'!")
                                    incorrect_input = False
                                else:
                                    confirm = input(f"The value {new_value} is not specified for this column. Are you sure that you want to replace {value} with it? (y/n): ")
                                    if confirm.lower().strip()[0] == 'y':
                                        for i in range(df.shape[0]):
                                            if df.iat[i,column_index] == value:
                                                df.iat[i,column_index] = new_value
                                        print(f"All values '{value}' replaced with '{new_value}'!")
                                        incorrect_input = False
                                    elif confirm.lower().strip()[0] == 'n':
                                        agr = input("Would you like to specify another value? (y/n): ")
                                        if agr.lower().strip()[0] == 'y':
                                            print("Repeat the process! \n")
                                        elif confirm.lower().strip()[0] == 'n':
                                            print(f"Values '{value}' were not replaced in {column}!")
                                            incorrect_input = False
                                        else:
                                            print("Incorrect input! start again! \n")
                                    else:
                                            print("Incorrect input! start again! \n")

                            elif agreed.lower().strip()[0] == 'n':
                                print(f"Values '{value}' were not replaced in {column}!")
                                incorrect_input = False
                            else:
                                print("Incorrect input! start again!")
        return df

# Merging databases

    #Merge two databases
    def binary_merge(self, df_1, df_2):
        err = self.check_names(df_1)
        if err == 0:
            err = self.check_names(df_2)
            if err == 0:
                unified = pd.concat([df_1,df_2])
                return unified
            else:
                print(f"The column names in Database 2 are incorrect! Please clean the databases before merging.")
        else:
            print(f"The column names in Database 1 are incorrect! Please clean the databases before merging.")

    # Merge arbitrary number of databases
    def merge(self, *args):
        i = 1
        correct_column_naming = True
        for df in args:
            err = self.check_names(df)
            if err == 0: 
                i+=1
            else:
                correct_column_naming = False
                print(f"The column names in Database {i} are incorrect! Please clean the databases before merging.")
                i+=1
        if correct_column_naming:
            unified = pd.concat(list(args))
            return unified
        else:
            print("Databases were not merged!")


#Comparing databses
    # Comparing predicates between databases. 
    def compare_predicates(self, df_1, df_2, pred: str):
        pred_1 = df_1.loc[df_1[f'{CORRECT_DESCRIPTIVE_COLUMN_NAMES[1]}'] == pred]
        pred_2 = df_2.loc[df_2[f'{CORRECT_DESCRIPTIVE_COLUMN_NAMES[1]}'] == pred]
        the_same = True
        for key in self.values_of_columns:
            if pred_1[f'{key}'].values[0] != pred_2[f'{key}'].values[0]:
                if pd.isna(pred_1[f'{key}'].values[0]) and pd.isna(pred_2[f'{key}'].values[0]):
                    pass
                else:
                    print(f"Predicte '{pred_1[CORRECT_DESCRIPTIVE_COLUMN_NAMES[0]].values[0]}' is {pred_1[key].values[0]} but predicate '{pred_2[CORRECT_DESCRIPTIVE_COLUMN_NAMES[0]].values[0]}' is {pred_2[key].values[0]} in column '{key}'")
                    the_same = False
        if the_same:
            print(f"Predicates '{pred_1[CORRECT_DESCRIPTIVE_COLUMN_NAMES[0]].values[0]}' and '{pred_2[CORRECT_DESCRIPTIVE_COLUMN_NAMES[0]].values[0]}' have the same properties!")
            return True
        else:
            return False

    #Comparing all predicatses between databases (to solve: two predicates in Database 2 with the same English translation)
    def compare_databases(self, df_1, df_2):
        not_in_1 = []
        not_in_2 = []
        different = []
        for predicate in df_1[f'{CORRECT_DESCRIPTIVE_COLUMN_NAMES[1]}']:
            if predicate in df_2[f'{CORRECT_DESCRIPTIVE_COLUMN_NAMES[1]}'].values:
                the_same = self.compare_predicates(df_1, df_2, predicate)
                if not the_same:
                    different.append(predicate)
            else:
                not_in_2.append(predicate)
        for predicate in df_2[f'{CORRECT_DESCRIPTIVE_COLUMN_NAMES[1]}']:
            if predicate not in df_1[f'{CORRECT_DESCRIPTIVE_COLUMN_NAMES[1]}'].values:
                not_in_1.append(predicate)
        if not_in_1 != []:
            print(f"The following predicates are absent in Database 1: {not_in_1}")
        if not_in_2 != []:
            print(f"The following predicates are absent in Database 2: {not_in_2}")
        if different != []:
            print(f"Database 1 and Database 2 differ on the following predicates: {different}.")
        if not_in_1 != not_in_2 and different == []:
            print("The common predicates of Databases 1 and Database 2 are the same!")
        elif different == []:
            print("All predicates in Database 1 and Databse 2 are the same!")
            return 0
        
        else:
            return 1





# Testing functions       
    def print_json(self, df):
        for column in df.columns[DESCRIPTIVE_COLUMNS:]:
            print(self.values_of_columns[column])


class AutomaticCleaner(DataCleaner):    
    def clean(self, df):
        clean = self.drop_empty_all(df)
        self.check_unnamed(clean)
        err = self.check_names(clean)
        
        clean = self.replace_nonalphanumeric(clean)
        clean = self.automatically_remove_typos(clean)
        if err == 0: 
            clean = self.fix_column_values(clean)
        else:
            print('Cannot fix_column_values, as column names do not match the template')
        return clean

class DataExtractor:
    def __init__(self):
        self.correct_column_names = []
        for item in CORRECT_DESCRIPTIVE_COLUMN_NAMES:
            self.correct_column_names.append(item)
        
        with open ('values_of_columns.json', 'r') as file:
            self.values_of_columns = json.load(file)

        for key in self.values_of_columns:
            self.correct_column_names.append(key)
        self.selectional_properties = ['finite indicative declerative clause',
       'finite subjunctive declerative clause',
       'Finite d-linked constituent interrogative clause',
       'finite non-d-linked constituent interrogative clause',
       'finite polar interrogative clause',
       'finite alternative interrogative clause',
       'Finite non-d-linked constituent interrogative clause with main clause syntax',
       'non-finite declarative clause without subject',
       'non-finite d-linked constituent interrogative without subj',
       'non-finite non-d-linked constituent interrogative without subj',
       'non-finite polar int without subj', 'non-finite alt int without subj','finite polar interrogative clause with main clause syntax',
       'concealed q', "Intransitive use"]

# Extracting parts of the database
    def get_semantic_mecore(self, df):
        sem=df
        sem['label'] = sem['ignorance/belief wrt int'].fillna('anti-rogative')
        for value in self.values_of_columns['ignorance/belief wrt int']:
            sem['label'] = sem['label'].replace(value, 'responsive')
        #Drop other columns which are not defined for both anti-rogatives and responsive
        sem = sem.drop(['ignorance/belief wrt int','gradability wrt int', 'Q-to-P veridicality', 'Q-to-P distributivity', 'P-to-Q distributivity'], axis =1) 
        #Drop rogative predicates
        sem = sem.dropna()
        print(sem['label'].value_counts())
        sem = sem.reset_index()
        sem = sem.drop('index', axis=1)
        return sem


    def get_binary_column_names(self, df, column_name:str):
        #Get values of the column i.
        values = list(df[column_name].value_counts().index)
        return [f'{column_name}_{value}' for value in values]



    def get_semantic_properties(self, df):
        semantic_df = df.drop(self.selectional_properties, axis=1)
        semantic_df = semantic_df.drop(CORRECT_DESCRIPTIVE_COLUMN_NAMES[1:], axis=1)
        return semantic_df
    
    def get_selectional_properties(self, df):
        selectional_df = df[self.selectional_properties]
        return selectional_df
    
    def get_anti_rogative_vs_responsive_df(self, df):
        semantic_df = self.get_semantic_properties(df)
         # Create label column from 'ignorance/belief wrt int'
        semantic_df['ignorance/belief wrt int'] = semantic_df['ignorance/belief wrt int'].fillna('anti-rogative')
        semantic_df['label'] = semantic_df['ignorance/belief wrt int']
        for value in self.values_of_columns['ignorance/belief wrt int']:
            semantic_df['label'] = semantic_df['label'].replace(value, 'responsive')
        #Check if the values are correct
        #Drop other columns which are not defined for both anti-rogatives and responsive
        semantic_df = semantic_df.drop(['gradability wrt int','Q-to-P veridicality', 'Q-to-P distributivity','P-to-Q distributivity', 'ignorance/belief wrt int'], axis =1) 
        #Drop rogative predicates
        semantic_df.to_csv('semantic.csv', index=False)
        semantic_df = semantic_df.dropna()
        print(semantic_df['label'].value_counts())
        semantic_df = semantic_df.reset_index()
        semantic_df = semantic_df.set_index('index')
        return semantic_df

    #Returns a database with dummy columns, which is an input to the hypothesis finder. If remove typically it changes all the values startig with 'typically' to 'neither'. 
    def get_X(self, df, cols:list = CORRECT_DESCRIPTIVE_COLUMN_NAMES, label :str = 'label', remove_typically = False):
        X= df.drop(cols + [label], axis=1)
        if remove_typically:
            for i in range(X.shape[0]): #iterate over rows
                for j in range(X.shape[1]): #iterate over columns
                    try:
                        if X.iat[i,j].split(' ')[0] == 'typically':
                            if len(X.iat[i,j].split(' ')) > 1:
                                X.iat[i,j] = 'neither'
                            else:
                                X.iat[i,j] = 'compatible'
                    except AttributeError:
                        pass
        X = pd.get_dummies(X)
        return X
    def remove_typically(self, df):
            for i in range(df.shape[0]): #iterate over rows
                for j in range(df.shape[1]): #iterate over columns
                    try:
                        if df.iat[i,j].split(' ')[0] == 'typically':
                            if len(X.iat[i,j].split(' ')) > 1:
                                df.iat[i,j] = 'neither'
                            else:
                                df.iat[i,j] = 'compatible'
                    except AttributeError:
                        pass
    
# Tree based method to analyse databases
class HypothesesFinder:
    def __init__(self):
        self.correct_column_names = []
        for item in CORRECT_DESCRIPTIVE_COLUMN_NAMES:
            self.correct_column_names.append(item)
        
        with open ('values_of_columns.json', 'r') as file:
            self.values_of_columns = json.load(file)
        
        self.n_nodes = None
        self.children_left = None
        self.children_right = None
        self.feature = None
        self.threshold = None
        self.impurity = None
        self.value = None

    # This function inspired by https://stackoverflow.com/questions/66297576/how-to-retrieve-the-full-branch-path-leading-to-each-leaf-node-of-a-sklearn-deci
    # Return the branches of a tree in a numerical form (list of lists of node indicies)
    #It is meant to be used as a part of retreive_text_branches(model, X)

    def retrieve_branches(self, number_nodes, children_left_list, children_right_list):
        """Retrieve decision tree branches"""
        
        # Calculate if a node is a leaf
        is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
        
        # Store the branches paths
        paths = []
        
        for i in range(number_nodes):
            if is_leaves_list[i]:
                # Search leaf node in previous paths
                end_node = [path[-1] for path in paths]

                # If it is a leave node yield the path
                if i in end_node:
                    output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                    yield output
                    
            else:
                
                # Origin and end nodes
                origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

                # Iterate over previous paths to add nodes
                for index, path in enumerate(paths):
                    if origin == path[-1]:
                        paths[index] = path + [end_l]
                        paths.append(path + [end_r])

                # Initialize path in first iteration
                if i == 0:
                    paths.append([i, self.children_left[i]])
                    paths.append([i, self.children_right[i]])

    # Returns branches of a tree in textual form: list of lists of pairs (property, threashold) ending with a pair (label, number of samples.) e.g. 
    #       [[('Complement projection/reversion through negation_neither', '> 0.5'),('Preference_incompatible', '<= 0.5'), ('responsive', 13)], ...]
    def retreive_text_branches(self, model, X, y, exception_size :int = 0, exception_indexes=False):
        self.n_nodes = model.tree_.node_count
        self.children_left = model.tree_.children_left
        self.children_right = model.tree_.children_right
        self.feature = model.tree_.feature
        self.threshold = model.tree_.threshold
        self.impurity = model.tree_.impurity
        self.value = model.tree_.value
        all_branches = list(self.retrieve_branches(self.n_nodes, self.children_left, self.children_right))
        class_names = model.classes_
        text_branches = []
        # Coinsider pure nodes only 
        if exception_size == 0:
            for branch in all_branches:
                #First find all the pure hypotheses consider pure leaves if exception_size = 0
                if model.tree_.impurity[branch[-1]] == 0.0:
                    textual = []
                    index=0
                    for node in branch: 
                        #For leaf nodes take the predicted label and the number of predicates 
                        if node == branch[-1]:
                            value = None
                            if model.tree_.n_outputs == 1:
                                value = model.tree_.value[node][0]
                            else:
                                value = model.tree_.value[node].T[0]
                            class_name = np.argmax(value)
                            if model.tree_.n_classes[0] != 1 and model.tree_.n_outputs == 1:
                                class_name = class_names[class_name]
                            textual.append((class_name, model.tree_.n_node_samples[node]))
                        #For decision nodes take the feature (property) on which it splits and the threshold value for that property. 
                        else:
                            if branch[index+1] == model.tree_.children_left[node]:
                                textual.append((X.columns[model.tree_.feature[node]], f"0"))
                            else:
                                textual.append((X.columns[model.tree_.feature[node]], f"1"))
                        index+=1
                    text_branches.append(textual)
            text_branches =  sorted(text_branches, key=lambda x: x[-1][-1],reverse=True)
        else:
            for branch in all_branches:
                textual = []
                index=0
                hypothesis_found = False
                for node in branch:
                    exc = int(sum(model.tree_.value[node][0]) - max(model.tree_.value[node][0])) # count exceptions
                    #print(model.tree_.value[node][0][0])
                    if exc <= exception_size:
                        hypothesis_found = True
                        value = None
                        if model.tree_.n_outputs == 1:
                            value = model.tree_.value[node][0]
                        else:
                            value = model.tree_.value[node].T[0]
                        class_name = np.argmax(value) #Returns the indices of the maximum values along an axis.
                        if model.tree_.n_classes[0] != 1 and model.tree_.n_outputs == 1:
                            class_name = class_names[class_name]
                        if exception_indexes:
                            exc_ind=[]
                            node_indicator = model.decision_path(X)
                            for x in X.index:
                                node_index = node_indicator.indices[
                                node_indicator.indptr[x] : node_indicator.indptr[x + 1]]
                                if node in node_index and y.iloc[x] != class_name:
                                    exc_ind.append(x)
                            textual.append((class_name, int(max(model.tree_.value[node][0])), f'exceptions: {exc}', f'exception_indexes: {exc_ind}'))
                        else:
                            textual.append((class_name, int(max(model.tree_.value[node][0])), f'exceptions: {exc}'))
                    #For decision nodes take the feature (property) on which it splits and the threshold value for that property. 
                    if node != branch[-1]:
                        if branch[index+1] == model.tree_.children_left[node]:
                            textual.append((X.columns[model.tree_.feature[node]], f"0"))
                        else:
                            textual.append((X.columns[model.tree_.feature[node]], f"1"))
                    index+=1
                if hypothesis_found:
                    text_branches.append(textual)

            text_branches =  sorted(text_branches, key=lambda x: x[-1][-2],reverse=True)
        return text_branches
    
    #Removes hypotheses that predict behaviour of less than LIMIT_OF_SAMPLES predicates
    def remove_overfitting(self, hypotheses):
        new_list = []
        for hypothesis in hypotheses:
            if hypothesis[-1][1] > LIMIT_OF_SAMPLES:
                new_list.append(hypothesis)
        return new_list
    
    #Investigate whether a set of predicates X (rows) satisfy a simple rule on label y defined by one of the properties (e.g. if all neg-raising predicates are anti-rogative):
    def individual_check(self, X, y, properties=None, printing=True):
        if properties == None:
            properties = X.columns
        hypotheses_list = []
        for properties in [list(properties)]:
            for property in properties:
                for value in X[property].value_counts().index:
                    predicates =[]
                    labels = []
                    for row in X.index:
                        if X[property].loc[row] == value:
                            try:
                             predicates.append(row)
                            except:
                                pass
                            labels.append(y.loc[row])
                    if len(labels) > LIMIT_OF_SAMPLES and len(set(labels)) == 1:
                        hypotheses_list.append({'hypothesis': (property, str(value)), 'label': str(labels[0]), 'predicates': predicates, 'column':y.name})
                        if printing:
                            print(f"All predicates ({len(labels)}) that has value: {str(value)} in property {property} are {str(labels[0])}")
        return hypotheses_list
    
    #Check if a conjunction of properties consitiute a hypothesis. If subsets=True checks also all the subsets of properties
    def conjunctive_check(self, X, y, properties: list, subsets=False, printing=True, exception_size :int= 0):
        results = False
        if len(properties) > MAXIMAL_NUMBER_OF_PROPERTIES:
            print(f"The maximal number of properties considered as a hypothesis is {MAXIMAL_NUMBER_OF_PROPERTIES}. Provide less properties!")
        else: 
            properties_with_values = []
            for property in properties:
                property_values_list = []
                try:
                    for value in X[property].value_counts().index:
                        property_values_list.append((property,value))
                    properties_with_values.append(property_values_list)
                except KeyError as e:
                    print(f"At least one of the properties ({str(e)}) you specified does not occur in the database!")
                    return e
            #Take all the possible combination of the values of the three conjoined properties with or without their subsets:
            hypotheses_list = []
            if not subsets: 
                for conjunction in itertools.product(*properties_with_values):
                    database = X
                    for conjunct in conjunction:
                        database = database[database[conjunct[0]] == conjunct[1]]
                        if database.empty:
                            break
                    labels=[]
                    predicates = []
                    for row in database.index:
                        try:
                            predicates.append(row)
                        except:
                            pass
                        labels.append(y.loc[row])
                    if len(labels) > LIMIT_OF_SAMPLES and len(set(labels)) == 1:
                        results = True
                        hypotheses_list.append({'hypothesis': conjunction, 'label': labels[0], 'predicates': predicates})
                        if printing:
                            print(f"All predicates ({len(labels)}) which satisfy the following conjunction: {conjunction} are {labels[0]} ")
            if subsets:
                conjunctions = list(itertools.product(*properties_with_values))
                for conjunction in conjunctions:
                    sublists=[]
                    for i in range(1,len(conjunction)+1):
                        # Generating all the sub lists
                        sublists += [list(j) for j in itertools.combinations(conjunction, i)]
                    #For each hypothesis take only those predicates that satisfy the antecedent
                    for hypothesis in sublists:
                        database = X
                        for conjunct in hypothesis:
                            database = database[database[conjunct[0]] == conjunct[1]]
                            #Exclude the hypotheses with falsified antecedent
                            if database.empty:
                                break
                        predicates = []
                        labels=[]
                        for row in database.index:
                            try:
                                predicates.append(row)
                            except:
                                pass
                            labels.append(y.loc[row])
                        #Check if there are more than LIMIT_OF_SAMPLES predicates that satisfy the antecedent and whether they all share the same selectional property. 
                        if len(labels) > LIMIT_OF_SAMPLES and len(set(labels)) == 1:
                            results = True
                            hypotheses_list.append({'hypothesis': conjunction, 'label': labels[0], 'predicates': predicates})
                            if printing:
                                print(f"All predicates ({len(labels)}) which satisfy the following conjunction: {hypothesis} are {labels[0]}")
            if results:
                return hypotheses_list
            else:
                if printing:
                    print("This conjunction does not yield any hypotheses")
                return []
        
    # Discover new hypotheses in a database using a (non-random) forest based method. You can investigate the discovered hypotheses looking at their decision trees, and checking conjunctively.
    #TO DO: make this predictive
    def forest_based_discovery(self, X, y, limit :int=MAXIMAL_NUMBER_OF_PROPERTIES, exception_size :int= 0, exception_indexes=False):
        # Check if limit is feasible.
        if limit == 0:
            print("You need to consider hypotheses of lenght at least 1")
            return []
        if limit == 1:
            print("This tool is not useful for hypotheses of lenght 1. Performing individual check.")
            output = self.individual_check(X, y)
            return output
        if limit > MAXIMAL_NUMBER_OF_PROPERTIES:
            print(f"The maximal number of properties considered as a hypothesis is {MAXIMAL_NUMBER_OF_PROPERTIES}. Provide less properties!")

        hypotheses = []

        # Take all the possible combinations of the properties of size limit
        for properties in list(itertools.combinations(X.columns, limit)):
            # Fit the data from the subtable containing only the selected properties to a decision tree model.
            model = DecisionTreeClassifier(max_depth=limit+1)
            model.fit(X[list(properties)],y)
            # Recover the branches of the tree from that model, which end with pure leafs. Remove hypotheses that explain less than LIMIT_OF_SAMPLES predicates.
            branches = self.retreive_text_branches(model, X[list(properties)], y, exception_size = exception_size, exception_indexes=exception_indexes)
            branches = self.remove_overfitting(branches)
            for branch in branches:
                if branch not in hypotheses:
                    hypotheses.append(branch)
        if exception_size == 0:
            hypotheses =  sorted(hypotheses, key=lambda x: x[-1][-1],reverse=True)
        else:
            hypotheses =  sorted(hypotheses, key=lambda x: x[-1][-2],reverse=True)
        return hypotheses
    
# TO DO: Look at the sets of predicates and display the hypotheses which explain the same set of predicates or have a hypothesis that explain their superset.

    # Find relations between semantic properties in a database to be used to eliminate redundant hypotheses.
    # Returns a dictionary with antecedents as keys and properties they imply as values e.g. "{('Certainty_always', 0)": [["Uncertainty_incompatible","0"]] ...}
    # This means that if a predicate does not always imply certeinty then it is not incompatible with uncertainty
    # Note that these relations are recovered from the database, and do not need to hold in general, we do that to reduce the number of redundant hypotheses discovered by other functions. 
    # You can look at them, but e.g. the fact that all likelihood inncompatible predicates are non neg-raising may be accidental and characteristic for Polish  database only. 
    def find_semantic_relations(self, X, printing=False):
        # Take each property as a label
        hash_table = {}
        for property in X.columns:
            if printing:
                print(str(property))
            #Take the database without that property
            database = X.drop(property, axis=1)
            #For each property(antecedent) check if implies the investigated property
            antecedents = self.individual_check(database, X[property], printing=printing)
            #Create a dictionary with antecedents as keys and properties they imply as values e.g. "{('Certainty_always', 0)": [["Uncertainty_incompatible","0"]] ...}
           
            for antecedent in antecedents:
                if "." in antecedent["label"]:
                    antecedent["label"] = str(antecedent["label"]).split('.')[0]
                try:
                    hash_table[f'{antecedent["hypothesis"]}'].append((str(antecedent["column"]), str(antecedent["label"])))
                except KeyError:
                    hash_table[f'{antecedent["hypothesis"]}'] = [(str(antecedent["column"]), str(antecedent["label"]))]
        return(hash_table)
    
    #Takes a list of hypotheses (output of retreive_text_branches), and a dictionary of semantic interdependencies (output of find_semantic_relations) and eliminates redundancies.
    # Return a list of hypotheses without simple redundancies. 
    # TO DO: transform the conjunctive hypothese into branches to use eliminative functions.
    def eliminate_redundant_hypotheses(self, hypotheses, implications: dict):
        eliminated_a_hypothesis = True
        if implications == {}:
            print("The list of indeperdencies is empty, the function did not change anything")
            return hypotheses
        # Run as long as a change was made in the last run.
        while eliminated_a_hypothesis:
            #Take a hypothesis
            before = len(hypotheses)
            for hypothesis in hypotheses:
                #Create all the hypotheses that the given hypothesis implies
                redundancies = []
                # Take each property from the antecedent
                for property in hypothesis[:-1]:
                    # Take each property that it implies
                    consequents =[]
                    try:
                        consequents = implications[f"{property}"]
                    except KeyError:
                        pass
                    # If it has some implications construct all the redundant hypotheses by replacing the property with its consequent: 
                    if consequents != []:
                        for consequent in consequents:
                            redundancy = []
                            for conjunct in hypothesis:
                                if conjunct == property:
                                    redundancy.append(consequent)
                                else:
                                    redundancy.append(conjunct)
                            redundancies.append(redundancy)
                # If some redundancies were found eliminate them from the list of hypotheses
                if redundancies != []:
                    before = len(hypotheses)
                    for redundancy in redundancies:
                        try:
                            hypotheses.remove(redundancy)
                        except ValueError or AttributeError:
                            pass
            after = len(hypotheses)
            if before == after:
                print(len(hypotheses))
                return hypotheses
            
    def eliminate_order_significance(self, hypotheses):
        non_redundant = []
        for hypothesis in hypotheses:
            if sorted(hypothesis[:-1]) in non_redundant:
                hypotheses = hypotheses.remove(hypothesis)
            else:
                non_redundant.append(sorted(hypothesis[:-1]))
        print(len(hypotheses))
        return hypotheses

    def conjunction_to_branch_translation(self, conjunction):
        branch = []
        conjunction = conjunction[0]
        for hypothesis in conjunction["hypothesis"]:
            name = hypothesis[0]
            value = str(hypothesis[1])
            if "." in value:
                value = value.split('.')[0]
            branch.append((name,value))
        branch.append((conjunction["label"], len(conjunction["predicates"])))
        return branch
   
    def pruning_based_discovery(self, df, X, y, limit :int=MAXIMAL_NUMBER_OF_PROPERTIES, exception_size :int= 0, exception_indexes=False):
        # Check if limit is feasible.
        if limit == 0:
            print("You need to consider hypotheses of lenght at least 1")
            return []
        if limit == 1:
            print("This tool is not useful for hypotheses of lenght 1. Performing individual check.")
            output = self.individual_check(X, y)
            return output
        if limit > MAXIMAL_NUMBER_OF_PROPERTIES:
            print(f"The maximal number of properties considered as a hypothesis is {MAXIMAL_NUMBER_OF_PROPERTIES}. Provide less properties!")

        #The pruning alghoritm
        found_something = True
        X_limitted = X
        hypos = []
        while found_something:
            model = DecisionTreeClassifier(max_depth= 4)
            model.fit(X_limitted,y)
            # Recover the branches of the tree from that model, which end with pure leafs. Remove hypotheses that explain less than LIMIT_OF_SAMPLES predicates.
            branches = self.retreive_text_branches(model, X_limitted, y, exception_size=exception_size, exception_indexes=exception_indexes)
            branches = self.remove_overfitting(branches)
            if branches == []:
                found_something = False
            for branch in branches:
                if branch not in hypos:
                    hypos.append(branch)
            # plt.figure(figsize=(12,8), dpi=200)
            # plot_tree(model, feature_names=X.columns, filled=True);
            X_limitted = X_limitted.drop(X_limitted.columns[model.tree_.feature[0]], axis=1)
        return sorted(hypos, key=lambda x: x[-1][1],reverse=True) 
    
    def sort_hyp(self, hypotheses, ordering = 'ratio'):
        if ordering == 'ratio':
            return sorted(hypotheses, key = lambda x: int(x[-1][-2].split(': ')[-1])/x[-1][-3])
        if ordering == 'min_ratio':
            for h in hypotheses:
                min_ratio = 1
                for x in h:
                    if x[-2].split(': ')[0] == 'exceptions':
                        ratio = int(x[-2].split(': ')[-1])/x[-3]
                        if ratio < min_ratio:
                            min_ratio = ratio
                h.append(min_ratio)
            return sorted(hypotheses, key = lambda x: x[-1])



        else: 
            print('Invalid sorting')
            return hypotheses

        
