from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import sys, os, csv
import matrix_factorization #local

"""
@INPUT
    df : data frame with the columns that should be one-hot encoded
    column_names : the names of the columns to consider for one-hot encoding
@OUTPUT
    df : data frame with selected columns converted from multisets to one-hot encodings
"""
def one_hot(df, column_names):
    for column in column_names:
        print(column)
        #lb = LabelBinarizer()
        #df = df.join(pd.DataFrame(lb.fit_transform(df.pop(column)),
        #                    columns=[column + ": " + str(x) for x in lb.classes_],
        #                    index=df.index))

        mlb = MultiLabelBinarizer()
        df = df.join(pd.DataFrame(mlb.fit_transform(df.pop(column)),
                          columns=[column + ": " + str(x) for x in mlb.classes_],
                          index=df.index))
    return df

"""
@INPUT:
    key : key to join files on.
    filenames : list of files to be joined together. All must have <key> as a named column.    
@OUTPUT:
    df : a dataframe with the data from all of the files in filenames. 
"""
def join_together(key, filenames):
    #1) read in all the files, and store one row for each key
    df_list = []
    for name in filenames:
        df = pd.read_csv(name)
        if key in df.columns:
            value = df.columns.drop(key)[0]
            print(value)
            df = df.groupby(key)[value].apply(list).reset_index() 
            df_list.append(df)
        else:
            print("No column named \"" + key + "\" in " + name)

    if len(df_list) == 0:
        print("No files to input")
        return
    
    #2) join all of the dataframes on key
    joined_df = df_list[0]
    if len(df_list) > 1:
        for i in range(len(df_list)-1):
            joined_df = joined_df.merge(df_list[i+1],on=key,how='outer')
    print(joined_df)
    return joined_df

"""
@INPUT:
    df : dataframe with columns to be collapsed, prefixed with attribute
    attribute : the common prefix of the attributes to be condensed
@OUTPUT:
    df : dataframe with columns condensed
"""
def collapse_attribute(df, attribute):
    columns = [x for x in df.columns if attribute + ": " in x]
    print(attribute, columns)
    prefix_length = len(attribute) + 2
    new_attribute = []
    for index,row in df.iterrows():
        new_value = []
        for column in columns:
            if row[column]: new_value.append(column[prefix_length:])
        new_attribute.append(new_value)
    df = df.drop(columns,axis=1)
    df[attribute] = new_attribute
    return df

"""
@INPUT:
    filename : the name of the file this data was read from
    df : dataframe containing the data in list form
    key : key to associate with each row
    attribute : attribute to be written out
@OUTPUT:
"""
def write_out_file(filename, df, key, attribute):
    df = df[[key, attribute]]
    output = [[key, attribute]]
    for index, row in df.iterrows():
        key_value = row[key]
        attribute_values = row[attribute]
        for attribute_value in attribute_values:
            output.append([key_value, attribute_value])
    outfile = "out_" + filename
    with open(outfile,mode='w') as f:
        writer = csv.writer(f)
        writer.writerows(output)

"""
@INPUT:
    files : list of files to read in
    key : name of key to identify entities across the files
    num_synthetic_keys : number of "individuals" to create on the other end
    K : number of 'feature vectors' to use in the decomposition
    iterations : number of iterations to use for the matrix factorization
    epsilon : privacy budget
    lambda_ : learning rate
"""
def generate_synthetic_files(files, key, num_synthetic_keys, K, iterations, epsilon, lambda_=0.01):
    df = join_together(key, files)
    colnames = df.columns.drop(key)
    df = one_hot(df, df.columns.drop(key))

    R = df[df.columns.drop(key)].values
    private_df = matrix_factorization.run_and_sample_als(R,num_synthetic_keys,K,iterations,lambda_,epsilon)
    private_df.columns = df.columns.drop(key)
    private_df[key] = [i+1 for i in range(private_df.shape[0])]

    #Collapse back down from one-hot to a single attribute
    for column in colnames:
        private_df = collapse_attribute(private_df,column)

    #For each of these attributes, expand out the list and write the resulting file
    for i in range(len(colnames)):
       write_out_file(files[i], private_df, key, colnames[i]) 
     
if __name__ == "__main__":
    files = ["test2.csv","test1.csv"]
    key = "key"
    generate_synthetic_files(files, key, 4, 2, 100, 1)
        










