import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
#For image plotting
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from PIL import Image
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

#!!!!!!!!!!
#WARNING!!!!TODO:!!!!!
#Likely does not work for set values right now. Pushing through the cat/numeric version for testing first.
#!!!!!!!!!!!

class FidesDataset(object):

    #Constructor sets default values for new data, but copies over values from the previous FidesDataset in
    # the case that we are running an approximation method.
    def __init__(self, synthetic_method="", fd=None, data=None):
        #If we have a string in, we must have approximated from a previous FidesDataset so let's copy accordingly
        if synthetic_method:
            self.synthetic_method = fd.synthetic_method + "\n" + synthetic_method
            #Categorical and numeric file information
            self.cat_numeric_file_path = fd.cat_numeric_file_path        #Always holds the original filepath
            self.cat_numeric_df = pd.DataFrame()                         #DataFrame read from cat_numeric_file_path
            self.cat_cols = fd.cat_cols                                  #Column names of those to be considered as categorical attributes
            self.num_cols = fd.num_cols                                  #Column names of those to be considered as numeric attributes
            #Set valued file information
            self.set_valued_file_paths = fd.set_valued_file_paths        #List of paths to set valued files (each key can appear multiple times)
            self.set_valued_dfs = []                                     #List of dataframes of set valued files
            self.key = fd.key                                            #Index of the column to be considered as a key in the set files
            self.set_cols = fd.set_cols                                  #Columns names of those to be considered as set valued attributes
            #Joined data
            self.data_to_use = data                                      #Full DataFrame representation of the data to be used (subset of data from files)

        #If we don't have any string in then we must just be reading in from data. Set default values accross the board
        else:
            self.synthetic_method = ""
            #Categorical and numeric file information
            self.cat_numeric_file_path = ""         #Path to standard rectangular file with categorical and numeric attributes
            self.cat_numeric_df = pd.DataFrame()    #DataFrame read from cat_numeric_file_path
            self.cat_cols = []                      #Column names of those to be considered as categorical attributes
            self.num_cols = []                      #Column names of those to be considered as numeric attributes
            #Set valued file information
            self.set_valued_file_paths = []         #List of paths to set valued files (each key can appear multiple times)
            self.set_valued_dfs = []                #List of dataframes of set valued files
            self.key = []                           #Index of the column to be considered as a key in the set files
            self.set_cols = []                      #Columns names of those to be considered as set valued attributes
            #Joined data
            self.data_to_use = pd.DataFrame()       #DataFrame to be approximated

    """
    Read in the file containing categorical and numeric attributes.
    @Input:
        file_path: path to file
        delim: delimiter used to tread the file, comma by default
    @Output:
    """
    def read_categorical_numeric_file(self, file_path, delim=","):
        if self.synthetic_method:
            print("!!!! " + self.synthetic_method + " already run, cannot read in from disk !!!!")
            return
        self.cat_numeric_file_path = file_path
        self.cat_numeric_df = pd.read_csv(file_path, delimiter=delim)

    """
    Load the categorical and numeric attributes directly
    @Input:
        file_path: path to file
        df: dataframe with categorical and numeric attributes
    @Output:
    """
    def set_categorical_numeric_file(self, df):
        if self.synthetic_method:
            print("!!!! " + self.synthetic_method + " already run, cannot overwrite data !!!!")
            return
        self.cat_numeric_df = df.copy()

    """
    Set the columns to be treated as categorical
    @Input:
        cat_cols: the list of categorical columns
    @Output:
    """
    def set_categorical_columns(self, cat_cols):
        if self.synthetic_method:
            print("!!!! " + self.synthetic_method + " already run, cannot reset categorical columns !!!!")
            return
        self.cat_cols = cat_cols

    """
    Set the columns to be treated as numeric
    @Input:
        num_cols: the list of numeric columns
    @Output:
    """
    def set_numeric_columns(self, num_cols):
        if self.synthetic_method:
            print("!!!! " + self.synthetic_method + " already run, cannot reset numeric columns !!!!")
            return
        self.num_cols = num_cols

    """
    Read in a file containing a set valued attribute.
    @Input:
        file_path: path to file
        key: the column containing the key value
        delim: delimiter used to tread the file, comma by default
    @Output:
    """
    def read_set_valued_file(self, file_path, key, delim=","):
        if self.synthetic_method:
            print("!!!! " + self.synthetic_method + " already run, cannot read data from disk !!!!")
            return
        if not self.key:
            self.key = [key]
        elif not self.key[0] == key:
            raise ValueError("This key does not match. Expected " + str(self.key[0]))

        self.set_valued_file_paths.append(file_path, delimiter=delim)
        df = pandas.read_csv(file_path, delimiter=delim)
        if not key in df.columns:
            raise ValueError("The key " + key + " cannot be found in the specified file")

        df = df.groupby(key)[value].apply(list).reset_index()
        self.set_valued_dfs.append(df)

    """
    Use the already loaded dataframes to create the desired dataframe to be approximated.
    @Input:
    @Output:
    """
    def create_data_to_use(self):
        if self.synthetic_method:
            print("!!!! " + self.synthetic_method + " already run, cannot overwrite data to use !!!!")
            return
        #First join together all of the set valued dataframes
        joined_set_df = pd.DataFrame()
        set_cols = []
        #Make sure we have some, otherwise proceed with the empty ones
        if self.set_valued_dfs:
            joined_set_df = self.set_valued_dfs[0]
            if len(set_valued_dfs) > 1: #make sure we have more to join
                for i in range(len(self.set_valued_dfs)-1):
                    joined_set_df = joined_df.merge(self.set_valued_dfs[i+1],on=key,how='outer')
            joined_set_df = joined_df.fillna('')
            set_cols = joined_set_df.columns.drop(key)
        cat_num_columns = self.key + self.num_cols + self.cat_cols
        self.set_cols = set_cols
        try:
            cat_num_df = self.cat_numeric_df[cat_num_columns]
        except:
            print("Specified columns were not found in the categorical and numeric data: " + str(cat_num_columns))
            return
        if self.key:
            self.data_to_use = cat_num_df.merge(joined_set_df,on=key,how='left')
        else:
            self.data_to_use = cat_num_df

    """
    Print the data to a file along with a summary of methods run and attribute properties
    @INPUT:
        label       = new label for naming files
        folder_path = path to folder. Defaults to the same folder the categorical/numeric data was read from,
                      and if there was none then the working directory.
    @OUTPUT:
    """
    def write(self, label, folder_path=""):
        if not folder_path:
            if self.cat_numeric_file_path:
                folder_path = os.path.dirname(os.path.realpath(self.cat_numeric_file_path))
        out_csv = os.path.join(folder_path, label + ".csv")
        self.data_to_use.to_csv(out_csv, index=False)
        out_summary = os.path.join(folder_path, label + "_summary.txt")
        file = open(out_summary,"w")
        file.write("Data approximated: " + self.cat_numeric_file_path + "\n")
        file.write("---------------------------\n")
        file.write("\tMethod used: " + self.synthetic_method + "\n")
        file.write("---------------------------\n")
        file.write("\tNumeric columns: " + str(self.num_cols) + "\n")
        for num_attribute in self.num_cols:
            file.write("\t\t" + num_attribute + " "
            + str(min(self.data_to_use[num_attribute])) + " "
            + str(max(self.data_to_use[num_attribute])) + "\n")
        file.write("---------------------------\n")
        file.write("\tCategorical columns: " + str(self.cat_cols) + "\n")
        file.write("---------------------------\n")
        file.close()

    """
    Print a split of the data to be used for classification testing
    @INPUT:
        test_size =     fraction of data to be included in test file
        suffix =        suffix for the training and test output files
        folder_path =   path to folder. Defaults to the same folder the categorical/numeric data was read from.
    @OUTPUT:
        train_out =     path to training data
        test_out =      path to test data
    """
    def write_train_test_split(self, test_size=0.2, suffix="", folder_path=""):
        if not folder_path:
            if self.cat_numeric_file_path:
                folder_path = os.path.dirname(os.path.realpath(self.cat_numeric_file_path))
        train_out = os.path.join(folder_path, "train" + suffix + ".csv")
        test_out = os.path.join(folder_path, "test" + suffix + ".csv")
        train_df, test_df = train_test_split(self.data_to_use, test_size = test_size)
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        return train_out, test_out
