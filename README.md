# MECORE-database-anaylsis-tools

### This project serves to automotically analyse the data collected in the MECORE Project. It allows to safely unify, merge and compare databases.

TO DO: Implementation of Machine learning tools \

## Input file format
This package operates on .csv files from the MECORE DATABASE. 
You can find the most up to date version of the database here: (https://wuegaki.ppls.ed.ac.uk/mecore/mecore-databases/)
Example .csv files are available with this file in the folder csv_files_070823


## Loading and saving databases.
1. Make sure that you have all the files you intend to use in .csv format, and in the same folder as the program (or you know the relative filepaths)
2. To load a database use: df = pd.read_csv('[FLIENAME].csv')
3. Make sure to saving changes (inplace syntax) while performing the cleaniang using df = cleaner.'[FUNCTION]'(df)
4. To save the database use: df.to_csv("cleaned_df.csv", index=False) 

## Cleaning

The **data_cleaning.py** file allows to pre-process the data and is tuned specifically for the purpose of the project. 

To use it you need **values_of_columns.json** which specifies the names of the columns (properties) containing data and the allowed values in each column. 
The names of the descriptive columns are specified in data_cleaning.py (predicate', 'English translation', 'predicate class').

**values_of_columns.json** can be created using function *values_from_all_columns()* form *ValuesExtractor()* 

To use the cleaner you can follow the steps shown in **main.py**. 

### *AutomaticCleaner()* has the following function:
AutomaticCleaner.clean(df), which deletes empty rows and columns, checks if the columns are named correctly and allows to easily change the names, corrects the typos automatically and allow to easily fix incorrect values manually in prompt.

### *DataCleaner()* has the following functions, which allow to perform cleaning step-by-step:

**Deletes empty rows and columns:** 

*DataCleaner.drop_empty_rows(df)* gets rid of empty rows.

*DataCleaner.drop_empty_columns(df)* gets rid of empty columns.

*DataCleaner.drop_empty_all(df)* gets rid of empty rows and columns.

**Checks whether the file contains unnamed columns**

*DataCleaner.check_unnamed(df)*

**Checks whether the columns are named correctly and allows to change them using prompt**

*DataCleaner.check_names(df)* If the columns are named correctly you will be prompted that this is the case. If they are named
incorrectly, then for each incorrect name yo will be asked if you want to replace it with the correct one. If you will not replace the names
you will not be able to correctly fix the values using *DataCleaner.fix_column_values(df)*.

**Removes annotation and typos. Brings all words to lowercase.**

*DataCleaner.replace_nonalphanumeric(df)* removes annotations and makes the job easier for the spell-checker.

*DataCleaner.automatically_remove_typos(df)* removes typos automatically. 

*DataCleaner.manually_remove_typos(df)* removes typos but asks for confirmation of every change.  

**Unify the values in each column with the specified values**

*DataCleaner.fix_column_values(df)* For each column you will be informed if there are any empty cells, corresponding to "not applicable" value in the database (e.g. P-to-Q distributivity for an anti-rogative predicate is empty). If there are any cells with values that are not specified in *values_of_columns.json* you will be prompted how many of them are in the column and asked if you want to replace *all* of them with some value. If you will answer yes you will see the specified values. You can copy-paste one of them to the prompt and the program will automatically replace all the incorrect values. If you type your custom value you will be asked if you really want to replace the old value with an non-specified one. Accepting will result in replacing the previous value with your input. You can also decline the change and the program will leave the original value in all the cells. If you are sure that you want the old value to be there the cells you can also edit *values_of_columns.json* manually and add the new value to the right column to avoid error in the future.

*DataCleaner.values_from_all_columns(df)* Allows to check if the values were replaced correctly. 


## Merging
The **data_cleaning.py** file allows to merge databases and is tuned to the project template.

### *AutomaticCleaner()* and *DataCleaner()* have the following functions:

*df_bin = DataCleaner.binary_merge(df_1,df_2)* allwes to safely merge two databases

*df_merged = DataCleaner.merge(df_1,df_2,df_3)* allows to safely merge arbitrary number of databses specified as args. 

## Comparing
The **data_cleaning.py** file allows to compare databases and is tuned to the project template.

### *AutomaticCleaner()* and *DataCleaner()* have the following functions:

*DataCleaner.compare_predicates(self, df_1, df_2, pred: str)*, takes two databases and compares two preicates which correspond to the same English verb *pred*. 

*DataCleaner.compare_databases(self, df_1, df_2)* compares all pairs of predicates in two databases, which have the same English translation. You are prompted what are the differences between the databses and which predicates are missing from each Database with respect to the other. (to solve: two predicates in Database 2 with the same English translation)

## Extraction
The **data_cleaning.py** file allows to extract predefined parts of the database.

### *DataExtractor()* has the following functions:
*df_semantic = get_semantic_properties(df)* extracts semantic properties from the database (based on column names). 

*df_selectional = get_selectional_properties(df)* extracts selectional properties from the database (based on column names). 
    
*df_r = get_anti_rogative_vs_responsive_df(df)* extracts only anti_rogative and responsive predicates and their semantic properties. Creates additional column called 'label' which contains values: '*responsive*' or '*anti-rogative*'

## Analysis
The **data_cleaning.py** file allows to verify hypotheses about the behaviour of clause-embedding predicates and automatically search for new hypotheses.

### *Hypothesis Finder()* has the following functions:

*conjunctive_check(X, y, properties: list, subsets=False, printing=True)* takes data frame (X) and a label (y) from the same database and a list of properties. Returns the hypotheses which arise from the conjunction of those properties. If subsets = True, then returns all the hypotheses from any combination of those properties.

*forest_based_discovery(X, y, limit :int=MAXIMAL_NUMBER_OF_PROPERTIES)* takes data frame (X) and a label (y) from the same database. For each subset of properties in X of size smaller than *limit* returns all the hypotheses that allow to predict values of more than *LIMIT OF SAMPLES* predicates in y. 

*find_semantic_relations(X)* Retuns the list of all the implications between semantic properties in X. i.e. with properties as keys and properties they imply as values e.g. "{('Certainty_always', 0)": [["Uncertainty_incompatible","0"]] ...}

*df = eliminate_redundant_hypotheses(hypotheses, implications: dict)* takes a set of hypotheses (output of *forest_based_discovery*) and a list of implications (output of *find_semantic_relations*) and eliminate hypotheses which are redundant according to the implications. WARNIING: only order matters! The eliminated hypotheses may provide better explanation then those which remain. 

*df=eliminate_order_significance(hypotheses)* takes a set of hypotheses (output of *forest_based_discovery*) and returns a list of hypotheses, which 

