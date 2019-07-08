#This file provides methods for evaluating properies of individual FidesDatasets, statistical or otherwise.
from FidesDataset import FidesDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from SyntheticPrivate import normalize_data
from PIL import Image




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MI Heatmap ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
HELPER METHOD
Calculate the pairwise mutual information between attributes
@Input:
    dataset: the dataset over which to calculate PMI
@Output:
    mi_df: dataframe containing the mutual information scores
"""
def pairwise_attributes_normalized_mutual_information(dataset):
    mi_df = pd.DataFrame(columns=dataset.columns, index=dataset.columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method = "arithmetic")
    return mi_df

"""
Create and either display or save a mutual information Heatmap
@Input:
    fd:             FidesDataset to be visualized
    filepath:       filepath on which to save the image
@Output:
"""
def mutual_information_heatmap(fd, label, folder="", show=False):
    if not folder:
        folder = os.path.dirname(os.path.realpath(fd.cat_numeric_file_path))
    file_path_to_save = os.path.join(folder, label + ".png")
    mi = pairwise_attributes_normalized_mutual_information(fd.data_to_use[fd.cat_cols])

    fig = plt.figure(figsize=(15, 6), dpi=120)
    fig.suptitle('Pairwise Mutual Information', fontsize=20)
    sns.heatmap(mi, cmap="YlGnBu_r")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(file_path_to_save)
    if show:
        plt.show()
    plt.close(fig)
    #plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Total MI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
HELPER METHOD
Calculate the pairwise mutual information between attributes
@Input:
    dataset: the dataset over which to calculate PMI
@Output:
    mi_df: dataframe containing the mutual information scores
"""
def pairwise_attributes_mutual_information(dataset):
    mi_df = pd.DataFrame(columns=dataset.columns, index=dataset.columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str))
    return mi_df

"""
Create and either display or save a mutual information Heatmap
@Input:
    fd:             FidesDataset to be visualized
@Output:
    total_mi:       Total mutual information between all non
"""
def total_mutual_information(fd, filepath=""):
    mi = pairwise_attributes_mutual_information(fd.data_to_use[fd.cat_cols])
    total_mi = np.triu(mi.values, 1).sum()
    return total_mi



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Save the data image from this FidesDataset
@Input:
    fd: the FidesDataset whose data is to be plotted
@Output:
"""

def plot_data(fd, label, folder_path=""):
    if not folder_path:
        folder_path = os.path.dirname(os.path.realpath(fd.cat_numeric_file_path))
    file_path_to_save = os.path.join(folder_path, label + ".png")
    data_to_plot, num_parameters, pivoted_columns = normalize_data(fd)
    data_to_plot = data_to_plot.clip(min=0,max=1)
    img = Image.fromarray(np.uint8(data_to_plot*255))
    img.save(file_path_to_save)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SVM Accuracy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
HELPER METHOD
take a fides dataset and split it into two new datasets, train and test
@Input:
    fd:                 FidesDataset to be split into train and test
    test_proportion:    proportion of data to include in test set
    suffix:             suffix to write on train/test files
    folder_path:        folder in which to put the data
@Output:
    train_fd:           FidesDatset with the training data
    test_fd:            FidesDatset with the test data
"""
def split_for_training(fd, test_proportion = 0.2, suffix="", folder_path=""):
    train_file, test_file = fd.write_train_test_split(test_proportion, suffix, folder_path)
    train_fd = FidesDataset()
    train_fd.read_categorical_numeric_file(train_file)
    train_fd.set_categorical_columns(fd.cat_cols)
    train_fd.set_numeric_columns(fd.num_cols)
    train_fd.create_data_to_use()
    test_fd = FidesDataset()
    test_fd.read_categorical_numeric_file(test_file)
    test_fd.set_categorical_columns(fd.cat_cols)
    test_fd.set_numeric_columns(fd.num_cols)
    test_fd.create_data_to_use()
    return train_fd, test_fd

"""
Train an SVM classifier and output its accuracy
@Input:
    train_fd:    FidesDataset with the training data
    test_fd:     FidesDataset with the test data
    outcome_col: Name of the column with the variable to be predicted
@Output:
    acc: accuracy of the classifier
"""
def svm_accuracy(train_fd, test_fd, outcome_col):
    #Get the dataframes from the FidesDatasets
    train_df = train_fd.data_to_use.drop([outcome_col], axis=1)
    test_df = test_fd.data_to_use.drop([outcome_col],axis=1)
    #Find the categorical and numeric columns to be used
    cat_cols = test_fd.cat_cols.copy()
    cat_cols.remove(outcome_col)
    num_cols = test_fd.num_cols
    #Convert categorical values to 1-hot encoding for classifier compatibility
    one_hot_encoder = preprocessing.OneHotEncoder()
    one_hot_encoder.fit(train_df[cat_cols].append(test_df[cat_cols]))
    train_cat_X_sparse = one_hot_encoder.transform(train_df[cat_cols])
    test_cat_X_sparse = one_hot_encoder.transform(test_df[cat_cols])
    #Combine the one-hot with numeric values
    train_X = np.append(train_df[num_cols].values, train_cat_X_sparse.todense(),axis=1)
    test_X = np.append(test_df[num_cols].values, test_cat_X_sparse.todense(),axis=1)
    #Get the outcome column
    train_y = train_fd.data_to_use[outcome_col]
    test_y = test_fd.data_to_use[outcome_col]
    #Train and run the classifier
    linearsvc = LinearSVC(tol=1e-5)
    try:
        linearsvc.fit(train_X, train_y)
        test_acc =linearsvc.score(test_X, test_y)
        train_acc = linearsvc.score(train_X, train_y)
    except ValueError as e:
        print(e)
        acc = 0
    return test_acc, train_acc

def svm_accuracy_tune_reg(train_fd, test_fd, outcome_col, C=1):
    #Get the dataframes from the FidesDatasets
    train_df = train_fd.data_to_use.drop([outcome_col], axis=1)
    test_df = test_fd.data_to_use.drop([outcome_col],axis=1)
    #Find the categorical and numeric columns to be used
    cat_cols = test_fd.cat_cols.copy()
    cat_cols.remove(outcome_col)
    num_cols = test_fd.num_cols
    #Convert categorical values to 1-hot encoding for classifier compatibility
    one_hot_encoder = preprocessing.OneHotEncoder()
    one_hot_encoder.fit(train_df[cat_cols].append(test_df[cat_cols]))
    train_cat_X_sparse = one_hot_encoder.transform(train_df[cat_cols])
    test_cat_X_sparse = one_hot_encoder.transform(test_df[cat_cols])
    #Combine the one-hot with numeric values
    train_X = np.append(train_df[num_cols].values, train_cat_X_sparse.todense(),axis=1)
    test_X = np.append(test_df[num_cols].values, test_cat_X_sparse.todense(),axis=1)
    #Get the outcome column
    train_y = train_fd.data_to_use[outcome_col]
    test_y = test_fd.data_to_use[outcome_col]
    #Train and run the classifier
    linearsvc = LinearSVC(tol=1e-5, C=C)
    try:
        linearsvc.fit(train_X, train_y)
        acc =linearsvc.score(test_X, test_y)
    except ValueError as e:
        print(e)
        acc = 0
    return acc


def logit_regression(train_fd, test_fd, outcome_col, C=1):
    #Get the dataframes from the FidesDatasets
    train_df = train_fd.data_to_use.drop([outcome_col], axis=1)
    test_df = test_fd.data_to_use.drop([outcome_col],axis=1)
    #Find the categorical and numeric columns to be used
    cat_cols = test_fd.cat_cols.copy()
    cat_cols.remove(outcome_col)
    num_cols = test_fd.num_cols
    #Convert categorical values to 1-hot encoding for classifier compatibility
    one_hot_encoder = preprocessing.OneHotEncoder()
    one_hot_encoder.fit(train_df[cat_cols].append(test_df[cat_cols]))
    train_cat_X_sparse = one_hot_encoder.transform(train_df[cat_cols])
    test_cat_X_sparse = one_hot_encoder.transform(test_df[cat_cols])
    #Combine the one-hot with numeric values
    train_X = np.append(train_df[num_cols].values, train_cat_X_sparse.todense(),axis=1)
    test_X = np.append(test_df[num_cols].values, test_cat_X_sparse.todense(),axis=1)
    #Get the outcome column
    train_y = train_fd.data_to_use[outcome_col]
    test_y = test_fd.data_to_use[outcome_col]
    #Train and run the classifier
    logit_reg = LogisticRegression(tol=1e-5,C=C)
    try:
        logit_reg.fit(train_X, train_y)
        acc =logit_reg.score(test_X, test_y)
    except ValueError as e:
        print(e)
        acc = 0
    coefficients = logit_reg.coef_
    return acc, coefficients
