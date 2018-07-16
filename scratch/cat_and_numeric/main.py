from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import sys, os, csv
sys.path.insert(0, '../one_hot_encoding/')
import matrix_factorization

"""
@INPUT
    df : data frame with the columns that should be one-hot encoded
@OUTPUT
    df : data frame with selected columns converted from multisets to one-hot encodings
"""
def cat_one_hot(df):
    columns = df.columns
    for column in columns:
        print('cat - ' + column)
        lb = LabelBinarizer()
        df = df.join(pd.DataFrame(lb.fit_transform(df.pop(column)),
                            columns=[column + ": " + str(x) for x in lb.classes_],
                            index=df.index))

        #mlb = MultiLabelBinarizer()
        #df = df.join(pd.DataFrame(mlb.fit_transform(df.pop(column)),
        #                  columns=[column + ": " + str(x) for x in mlb.classes_],
        #                  index=df.index))
    return df

def set_one_hot(df, columns):
    for column in columns:
        print('set - ' + column)
        mlb = MultiLabelBinarizer()
        df = df.join(pd.DataFrame(mlb.fit_transform(df.pop(column)),
                          columns=[column + ": " + str(x) for x in mlb.classes_],
                          index=df.index))
    return df
"""
@INPUT
    df : data frame with only numeric columns
@OUTPUT
    df : normalized data
    stat_dict : mins and maxes for every column
"""
def normalize_numeric(df):
    stat_dict = dict()
    for column in df.columns:
        print('num - ' + column)
        stat_dict[column] = dict()
        col_min = np.min(df[column])
        col_max = np.max(df[column])
        df[column] = [(x-float(col_min)) / (float(col_max) - float(col_min)) for x in df[column]]
        stat_dict[column]["min"] = col_min
        stat_dict[column]["max"] = col_max
    return df, stat_dict 

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
    column : column to be rehydrated
    col_max : maximum value
    col_min : minimum value
@OUTPUT:
    new_column : rehydrated column
"""
def rehydrate_numeric(column, col_max, col_min):
    return [x*(col_max-col_min)+col_min for x in column]


"""
@INPUT:
    R : original values from the one-hot encoding subsetted to the set-valued columns
    R_hat : approximation of R for the same columns
    epsilon : privacy budget to be used for approximating the number of 1s.
@OUTPUT:
    new_columns:
"""
def approximate_one_zeros(R, R_hat, epsilon=0):
    num_non_zero = round(np.random.laplace(np.sum(R),1/(epsilon))) if epsilon else np.sum(R)
    if num_non_zero > R.shape[0]*R.shape[1]: num_non_zero = R.shape[0]*R.shape[1]
    elif num_non_zero < 0: num_non_zero = 1
    threshold = np.partition(R_hat.flatten(), -num_non_zero)[-num_non_zero]
    output = R_hat>=threshold
    return output 

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
def generate_synthetic_files(cat_num_file, set_valued_files, key, cat_cols, num_cols, num_synthetic_keys, K, iterations, epsilon, lambda_=0.01):
    df = pd.read_csv(cat_num_file)
    print(df)

    cat_df = df[cat_cols].copy()
    num_df = df[num_cols].copy()

    #binarize cat
    cat_df = cat_one_hot(cat_df)

    #normalize numeric
    num_df,stat_map = normalize_numeric(num_df)

    #now deal with the set-valued files
    set_df = join_together(key, set_valued_files)
    set_cols = set_df.columns.drop(key)
    set_df = set_one_hot(set_df, set_cols).drop(key,axis=1)

    #put these together into a new dataframe and get the values all at once
    full_df = pd.concat((num_df,cat_df,set_df),axis=1)
    print(full_df.columns)
    
    R = full_df.values 
    
    #calculate the approximation
    R_hat = matrix_factorization.run_als(R,K,iterations,lambda_,epsilon*0.99)

    #Unpack the results
    new_df = pd.DataFrame()

    col_counter = 0 #keeps track of how many columns we've used for simplicity later
    #Numeric - rehydrate according to previous min and max
    for i in range(len(num_cols)):
        new_col = rehydrate_numeric(R_hat[:,i],stat_map[num_cols[i]]['max'],stat_map[num_cols[i]]['min'])
        new_df[num_cols[i]] = new_col
        col_counter += 1
        
    #Categorical - choose the largest value among the matrix (closest to one)
    for cat_attribute in cat_cols:
        pivoted_columns = [x[len(cat_attribute)+2:] for x in full_df.columns if cat_attribute == x[:len(cat_attribute)]]
        pivoted_column_indices = [full_df.columns.get_loc(cat_attribute + ": " + x) for x in pivoted_columns]
        new_col = [pivoted_columns[np.argmax(R_hat[i,pivoted_column_indices])] for i in range(R.shape[0])]     
        new_df[cat_attribute] = new_col
        col_counter += len(pivoted_columns)

    #Set valued - use the rest of the privacy budget to choose among the rows in R
    #here's where the col_counter comes in, everything else in R should be for the set_valued.
    new_one_zeros = approximate_one_zeros(R[:,col_counter:],R_hat[:,col_counter:],0.01*epsilon)
    new_one_zeros = new_one_zeros.astype(int)
    new_col_counter = 0
    for set_attribute in set_cols:
        pivoted_columns = [x[len(set_attribute)+2:] for x in full_df.columns if set_attribute == x.split(":")[0]]
        pivoted_column_indices = [full_df.columns.get_loc(set_attribute + ": " + x) for x in pivoted_columns]
        new_col = []
        for i in range(R.shape[0]):
            row_col = []
            for index in np.nonzero(new_one_zeros[i,new_col_counter:new_col_counter+len(pivoted_columns)])[0]:
                print(index)
                row_col.append(pivoted_columns[index])
            new_col.append(row_col)
        new_col_counter += len(pivoted_columns)
        new_df[set_attribute] = new_col

    print(new_df)

    new_df.to_csv(cat_num_file[:-4] + "_synthetic.csv")
    

    
if __name__ == "__main__":
    cat_numeric_filename = "test.csv"
    set_valued_files = ["../one_hot_encoding/test1.csv","../one_hot_encoding/test2.csv"]
    key = "key"
    cat_cols = ["d","b"]
    num_cols = ["a","c","e"]
    generate_synthetic_files(cat_numeric_filename, set_valued_files, key, cat_cols, num_cols, 4, 2, 100, 100)
        









