#This file provides methods for evaluating the utility of approximated FidesDatasets by comparing two to each other.
import pandas as pd
import sys
import itertools
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~ Average Variation Distance across Marginals ~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Compare the variation distance between marginal counts over categorical attributes
@Input:
    fd:             FidesDataset to be visualized
@Output:
"""
def compare_marginals(original_fd, modified_fd, marginal_size):
    cat_cols = original_fd.cat_cols
    num_cols = len(cat_cols)
    num_marginals = len(list(itertools.combinations(cat_cols, marginal_size)))
    average_distance = 0
    for combo in itertools.combinations(cat_cols, marginal_size):
        old_series = original_fd.data_to_use[cat_cols].groupby(combo).size()/original_fd.data_to_use.shape[0]
        new_series = modified_fd.data_to_use[cat_cols].groupby(combo).size()/modified_fd.data_to_use.shape[0]
        difference = old_series - new_series
        #account for missing values
        for name, group in difference.iteritems():
            if np.isnan(group):
                if name in old_series.keys():
                    difference[name] = old_series[name]
                else:
                    difference[name] = new_series[name]
        l1_distance = sum(abs(difference))
        average_distance += l1_distance / num_marginals
    return(average_distance/2)
