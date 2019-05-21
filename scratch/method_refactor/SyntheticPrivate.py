#This file provides methods that take a FidesDataset and epsilon as input paramters (along with others, potentially) and
# outputs an approximated version of the dataset that satisfies epsilon-differential privacy
from FidesDataset import FidesDataset
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import NMF
import networkx as nx
from sklearn.metrics import mutual_info_score
from itertools import combinations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MF Method ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
HELPER METHOD
Transform the data_to_use dataframe from the FidesDatset to an array whose values are all between 0 and 1
@Input:
    fd:             FidesDataset to be approximated
@Output:
    R_values:       matrix representation of the data from fd
    num_parameters: dictionary with values for numeric columns
"""
def normalize_data(fd):
    R = pd.DataFrame()
    #NOTE: This will change significantly once I support attribute types properly
    num_parameters = dict()
    for num_attribute in fd.num_cols:
        num_parameters[num_attribute] = dict()
        col_min = np.min(fd.data_to_use[num_attribute])
        col_max = np.max(fd.data_to_use[num_attribute])
        if col_min == col_max: #need to catch numeric columns with only 1 entry
            R[num_attribute] = np.ones(len(fd.data_to_use[num_attribute]))
            num_parameters[num_attribute]["min"] = 0
            num_parameters[num_attribute]["max"] = col_max
        else:
            R[num_attribute] = [(x-float(col_min)) / (float(col_max) - float(col_min)) for x in fd.data_to_use[num_attribute]]
            num_parameters[num_attribute]["min"] = col_min
            num_parameters[num_attribute]["max"] = col_max
    for cat_attribute in fd.cat_cols:
        lb = LabelBinarizer()
        y = fd.data_to_use[cat_attribute]
        y = pd.get_dummies(y)
        y.columns = [cat_attribute + ": " + str(x) for x in y.columns]
        R = R.join(y,how='right')
    R = R.drop(fd.key,axis=1).fillna(0)
    pivoted_columns = R.columns
    R_values = R.values
    return R_values, num_parameters, pivoted_columns

"""
HELPER METHOD
@INPUT:
    column :     column to be rehydrated
    col_max :    maximum value
    col_min :    minimum value
@OUTPUT:
    new_column : rehydrated column
"""
def rehydrate_numeric(column, col_max, col_min):
    return [x*(col_max-col_min)+col_min for x in column]

"""
HELPER METHOD
"Rehydrate" the approximated matrix to create an approximated DataFrame
@INPUT:
    fd:               FidesDataset being approximated
    R:                numeric values to be rehydrated into a DataFrame
    num_parameters:   dictionary with values for numeric columns
@OUTPUT:
    reconstructed_df: approximated version of the numeric data from R
"""
def reconstruct_data(fd, R, num_parameters, piv_cols):
    reconstructed_df = pd.DataFrame()
    col_counter = 0 #keeps track of how many columns we've used for simplicity later
    #Numeric - rehydrate according to previous min and max
    for i in range(len(fd.num_cols)):
        new_col = rehydrate_numeric(R[:,i],num_parameters[fd.num_cols[i]]['max'],num_parameters[fd.num_cols[i]]['min'])
        reconstructed_df[fd.num_cols[i]] = new_col
        col_counter += 1
    #Categorical - choose the largest value among the matrix (closest to one)
    #TODO: Right now this will break if one column name is a subset of another!
    for cat_attribute in fd.cat_cols:
        pivoted_columns = [x[len(cat_attribute)+2:] for x in piv_cols if cat_attribute == x[:len(cat_attribute)]]
        pivoted_column_indices = [piv_cols.get_loc(cat_attribute + ": " + x) for x in pivoted_columns]
        new_col = [pivoted_columns[np.argmin(abs(R[i,pivoted_column_indices]-1))] for i in range(R.shape[0])]
        reconstructed_df[cat_attribute] = new_col
        col_counter += len(pivoted_columns)
    #Set valued - TODO: Ignoring for now
    return reconstructed_df

"""
PARENT METHOD
Matrix approximation
@INPUT:
    fd:             FidesDatset being approximated
    epsilon:        privacy budget
    k:              dimension of approximation
    num_iterations: number of iterations to run
    lambda_:        learning rate
@OUTPUT:
    new_fd:         FidesDataset with approximated data
"""
def matrix_factorization(fd, epsilon, k=100, num_iterations = 1000, lambda_ = 0.01, normalization="conditional"):
    #Get the matrix version of the dataset
    R, num_parameters, pivoted_columns = normalize_data(fd)
    #Run a matrix factorization approximation from SKLearn
    m, n = R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)
    noise = np.random.laplace(0,2*n/epsilon,size=(k, n))
    model = NMF(n_components=k, init='random', solver='mu', max_iter=num_iterations) #can set random_state=0 for debugging
    X = model.fit_transform(R)
    #Normalize rows
    if normalization == "conditional":
        norms = np.linalg.norm(X,ord=1,axis=1)
        for x in range(m):
            if norms[x] >0:
                X[x,:] = X[x,:]/norms[x]
    elif normalization == "all":
        norms = np.linalg.norm(X,ord=1,axis=1)
        for x in range(m):
            X[x,:] = X[x,:]/norms[x]
    elif normalization == "matrix":
        norm = np.linalg.norm(X)
        for x in range(m):
            X[x,:] = X[x,:]/norm
    #Add noise to result for Y
    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k),
                    np.dot(X.T, R) - noise/2)
    #Compute an approximate version of R
    approx_R = np.dot(X, Y)
    #Use the approximation to get an approximated DataFrame
    approx_df = reconstruct_data(fd, approx_R, num_parameters, pivoted_columns)
    method_summary = "Matrix Factorization: " + str(epsilon) + " " + str(k) + " " + str(num_iterations) + " " + str(lambda_)
    new_fd = FidesDataset(method_summary, fd, approx_df)
    return new_fd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~ Sampling-based Inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
HELPER METHOD
Calculate amplified privacy parameter for sampling
@INPUT:
    epsilon:    the original privacy budget
    n_sample:   the sample size
    n_dataset:  the number of tuples in the dataset
@OUTPUT:
    epsilon_a:  the amplified privacy budget
"""
def amplified_privacy_budget(epsilon, n_sample, n_dataset):
    epsilon_a = np.log2(np.exp(epsilon) - 1 + n_sample/n_dataset) - np.log2(n_sample/n_dataset)
    return epsilon_a

"""
HELPER METHOD
Calculate the global sensitivity of the mutual information function
@INPUT:
    n_sample:   the sample size (number of tuples)
@OUTPUT:
    delta_I:    the sensitivity of the mutual information function
"""
def mi_global_sensitivity(n_sample):
    delta_I = 2.0/n_sample * np.log2((n_sample+1)/2.0) + (n_sample-1)/n_sample * np.log2((n_sample+1)/(n_sample-1))
    return delta_I

"""
HELPER METHOD
Calculate the cutoff threshold for MI-based correlation
@INPUT:
    size_k:     the size of the domain of attribute k
    size_l:     the size of the domain of attribute l
    phi_c:      Cramer's V - default of 0.2 corresponds to weak dependency
@OUTPUT:
    threshold:  the cutoff value at which mutual information indicates an edge dependency
"""
def calculate_threshold(size_k, size_l, phi_c=0.2):
    threshold = (min(size_k, size_l)-1) * phi_c ** 2 / 2.0
    return threshold

"""
HELPER METHOD
Find the optimal sample size by minimizing the ratio delta_I/epsilon_a
@INPUT:
    epsilon:        the original privacy budget
    n_dataset:      the number of tuples in the dataset
@OUTPUT:
    optimal_n:      the optimal sample size to minimize the desired ratio
"""
def find_optimal_n_sample(epsilon, n_dataset):
    #Have to add 2 to avoid division by 0
    ratio_values = [mi_global_sensitivity(n_sample)/amplified_privacy_budget(epsilon, n_sample, n_dataset) for n_sample in range(2, n_dataset+1)]
    optimal_n = np.argmin(ratio_values)+2
    return optimal_n

"""
HELPER METHOD
Initialize the dependency graph
@INPUT:
    epsilon:        original privacy budget
    df:             original dataset
    phi_c:          Cramer's V - default of 0.2 corresponds to weak dependency
@OUTPUT:
    G:              Dependecy graph constructed by sampling the original df
"""
def initialize_dependecy_graph(epsilon, df, phi_c = 0.2):
    #Initialize the graph and add the attributes as nodes
    G = nx.Graph()
    G.add_nodes_from(df.columns)
    #Calculate the sampling rate
    n_dataset = df.shape[0]
    n_sample = find_optimal_n_sample(epsilon,n_dataset)
    sample_rate = n_sample/n_dataset
    #Get a sampled dataset from df
    df_sampled = df.sample(frac = sample_rate)
    #Calculate parameters for given n_sample
    epsilon_a = amplified_privacy_budget(epsilon, n_sample, n_dataset)
    mi_sensitivity = mi_global_sensitivity(n_sample)
    eta = np.random.laplace(loc = 0,scale = (2*mi_sensitivity)/epsilon_a)
    domain_sizes = {attribute: df[attribute].nunique() for attribute in df}
    attributes = list(df.columns)
    #Loop over unique attribute pairs and add edges as appropriate
    for k in range(len(attributes)):
        for l in range(k+1, len(attributes)):
            attribute_k = attributes[k]
            attribute_l = attributes[l]
            #Calculate the mutual information in the sampled dataset
            mi = mutual_info_score(df_sampled[attribute_k], df_sampled[attribute_l])
            noised_mi = mi + np.random.laplace(loc = 0,scale = (2*mi_sensitivity)/epsilon_a)
            #Calculate the perturbed threshold to be used
            threshold = calculate_threshold(domain_sizes[attribute_k], domain_sizes[attribute_l], phi_c)
            noised_threshold = threshold + eta
            #Add an edge if the mutual information exceeds the threshold
            if noised_mi >= noised_threshold:
                print("ADDING EDGE: ", attribute_k, attribute_l)
                G.add_edge(attribute_k, attribute_l)
    return G

"""
HELPER METHOD
triangulate a graph (make sure no cycle of length > 4 has disconnected non-adjacent vertices)
@INPUT:
    G:              graph to be triangulated
@OUTPUT:
    G_triangulated: triangulated version of G
"""
#OLD VERSION: We can pass a deletion sequence here, but I don't think we will every want to. Heuristic is better!
# def triangulate_graph(G, sequence=None):
#     if sequence is None:
#         sequence = np.random.permutation(G.nodes())
#     print(sequence)
#     H = G.copy()
#     G_triangulated = G.copy()
#     for node in sequence:
#         neighbors = H.neighbors(node)
#         pairs = list(combinations(neighbors, 2))
#         G_triangulated.add_edges_from(pairs)
#         H.add_edges_from(pairs)
#         H.remove_node(node)
#     return G_triangulated
#NOTE: there are heuristics that tend to perform better, but they are more complicated and probably not important for our implementation.
def triangulate_graph(G):
    H = G.copy()
    G_triangulated = G.copy()
    while len(H) > 0:
        degree_dict = H.degree()
        node = min(degree_dict, key=degree_dict.get)
        neighbors = H.neighbors(node)
        pairs = list(combinations(neighbors, 2))
        G_triangulated.add_edges_from(pairs)
        H.add_edges_from(pairs)
        H.remove_node(node)
    return G_triangulated

"""
HELPER METHOD
build the junction tree from the input graph G
@INPUT:
    G:              graph whose junction tree we want to output
@OUTPUT:
    junction_tree:  junction tree of G
"""
def get_junction_tree(G):
    G_triangulated = triangulate_graph(G)
    cliques = nx.find_cliques(G_triangulated)
    junction_tree = nx.Graph()
    junction_tree.add_nodes_from([tuple(clique) for clique in cliques])
    intersection_size = {combo:len(combo[0] + combo[1]) - len(set(combo[0] + combo[1])) for combo in combinations(junction_tree.nodes(), 2)}
    intersection_size = {key:val for key, val in intersection_size.items() if val != 0}
    ordered_by_weights = {pair[0] for pair in sorted(intersection_size.items(), key=lambda kv: -kv[1])}
    for pair in ordered_by_weights:
        cluster_1 = pair[0]
        cluster_2 = pair[1]
        if not nx.node_connected_component(junction_tree, cluster_1) == nx.node_connected_component(junction_tree, cluster_2):
            junction_tree.add_edge(cluster_1, cluster_2)
    return junction_tree

"""
HELPER METHOD
TODO: make this actually based off of the data
construct the cluster marginals based on the original data
@INPUT:
    jt:     the previously calculated junction tree
    data:   data to calculate marginals over
    epsilon: privacy budget
@OUTPUT:
    marginals: the consistent marginal dictionary
"""
def build_marginals(jt, df, epsilon):
    clusters = jt.nodes()
    separators = [set(e[0]).intersection(set(e[1])) for e in jt.edges()]
    separators.sort(key=len)
    marginal_tables = dict()
    #Build the marginal table for each cluster
    for cluster in clusters:
        marginal = df.groupby(cluster).size().reset_index(name='counts')
        marginal['counts'] += np.random.laplace(loc = 0,scale = (2*len(clusters))/epsilon,size=marginal.shape[0])
        counts = marginal._get_numeric_data()
        counts[counts < 0] = 0
        marginal['counts'] = counts
        marginal_tables[cluster] = marginal
    domain_sizes = {attribute: df[attribute].nunique() for attribute in df}
    #For each separator in sorted order by size (topographic set inclusion), we balance the cluster marginals
    for separator in separators:
        #Get the variance adjustment factors sigma_square
        sigma_squares = dict()
        for cluster in clusters:
            if all(elem in cluster for elem in separator):
                temp_cluster = list(cluster)
                for attribute in separator:
                    temp_cluster.remove(attribute)
                domains = [domain_sizes[attribute] for attribute in temp_cluster]
                sigma_square = np.product(domains)
                sigma_squares[cluster] = sigma_square
        #Now subset the data so that we can iterate over unique values in our separator attribute set
        separator_df = df.loc[:,separator]
        separator_df = separator_df.drop_duplicates()
        for index, row in separator_df.iterrows():
            cluster_marginals = dict()
            values = tuple(row)
            #Calculate the count of this value within each cluster that contains it
            numerator = 0
            denominator = 0
            for cluster in clusters:
                if all(elem in cluster for elem in separator):
                    temp_marginal = marginal_tables[cluster].copy()
                    for column in separator:
                        value = row[column]
                        temp_marginal = temp_marginal.loc[temp_marginal[column] == value]
                    cluster_marginals[cluster] = sum(temp_marginal['counts'])
                    numerator += cluster_marginals[cluster] / sigma_squares[cluster]
                    denominator += 1 / sigma_squares[cluster]
            overall_count = numerator / denominator
            #Now adjust all clusters in order to be in compliance with the overall count.
            for cluster in clusters:
                if all(elem in cluster for elem in separator):
                    adjustment = (overall_count - cluster_marginals[cluster])/sigma_squares[cluster]
                    mt = marginal_tables[cluster]
                    check = mt.loc[:,list(separator)] == row
                    rows_to_adjust = check.all(axis=1)
                    indices = mt[rows_to_adjust].index
                    marginal_tables[cluster].loc[indices,'counts'] += adjustment
    #Now that we have adjusted to be consistent across separators, we need to threshold them so that the counts sum to N
    #TODO: ^^ does that really work? I feel like that would break consistency pretty quickly, because you can't exactly do this over the whole dataset?
    for cluster in marginal_tables:
        counts = marginal_tables[cluster]._get_numeric_data()
        counts[counts < 0] = 0
        marginal_tables[cluster]['counts'] = counts
    return marginal_tables

"""
HELPER METHOD
Generate synthetic data from the noisy marginals
@INPUT:
    attributes: the columns for which we need to generate data
    jt:         the previously calculated junction tree
    marginals:  the marginals from which to sample
@OUTPUT:
    synthetic_df: the generated synthetic dataframe
"""
def generate_synthetic_from_marginals(attributes, jt, marginals, num_rows):
    clusters = set(jt.nodes())
    visited_clusters = set()
    sampled_attributes = set()
    synthetic_df = pd.DataFrame()
    #Get a random initial cluster and samle data from it
    while not visited_clusters == clusters:
        #Choose the next cluster to be sampled from
        neighbors = set()
        for cluster in visited_clusters:
            neighbors = neighbors.union(jt[cluster].keys())
        for cluster in visited_clusters:
            neighbors.discard(cluster)
        if not neighbors: #visited_clusters or neighbors are empty, choose one at random from what's left
            to_choose = clusters.copy()
            for cluster in visited_clusters:
                to_choose.remove(cluster)
            cluster = to_choose.pop()
        else: #Choose one of the neighbors at random
            cluster = neighbors.pop()
        print(cluster)
        #Check which attributes in this cluster we have already sampled
        cluster_attributes = set(cluster)
        already_sampled_attributes = cluster_attributes.intersection(sampled_attributes)
        if not already_sampled_attributes == cluster_attributes: #If they are equal, we don't have to do anything
            already_sampled_attributes = list(already_sampled_attributes)
            if synthetic_df.empty: #if no synthetic data yet, we can just sample directly from the marginal
                probs = marginals[cluster].counts / sum(marginals[cluster].counts)
                indices = [np.random.choice(marginals[cluster].index,p=probs) for i in range(num_rows)] #TODO: fix to number of datapoints to be sampled
                cluster_data = marginals[cluster].drop(['counts'], axis=1).loc[indices]
            else: #But if there is synthetic data already, we need to sample from the conditional distribution of this marginal.
                sampled_view = synthetic_df[already_sampled_attributes]
                samples_to_add = []
                #Store some dicts of what we've seen to speed up computation
                previously_seen_indices = dict()
                previously_seen_probs = dict()
                for index, row in sampled_view.iterrows():
                    tuple_version = tuple(element for element in row)
                    if tuple_version in previously_seen_probs:
                        indices = previously_seen_indices[tuple_version]
                        probs = previously_seen_probs[tuple_version]
                    else:
                        marginal = marginals[cluster].copy()
                        check = marginal.loc[:,already_sampled_attributes] == row
                        rows_to_sample = check.all(axis=1)
                        indices = marginal[rows_to_sample].index
                        probs = marginal.loc[indices].counts / sum(marginal.loc[indices].counts)
                        previously_seen_indices[tuple_version] = indices
                        previously_seen_probs[tuple_version] = probs
                    if len(indices) > 0: #We have to make sure there are actually some of these, noisy might mean there's not
                        samples_to_add.append(np.random.choice(indices, p=probs))
                    else: #if there aren't any, lets just add a row at random
                        print("Found the error and caught it")
                        indices = list(marginal.index)
                        samples_to_add.append(np.random.choice(indices, p=[1/len(indices) for elem in indices]))
                cluster_data = marginals[cluster].drop(['counts'], axis=1).loc[samples_to_add]
            #Now actually add this data to the synthetic dataframe
            for column in cluster_data:
                if not column in synthetic_df:
                    synthetic_df[column] = list(cluster_data[column])
        #Make sure we track what we have seen
        visited_clusters.add(cluster)
        for attr in cluster:
            sampled_attributes.add(attr)
    return synthetic_df

"""
Run the Sampling-Based Inference code and return a new FidesDatset
NOTE! This can only run over categorical variables.
@INPUT:
    fd:             FidesDatset being approximated
    epsilon:        privacy budget
    k:              dimension of approximation
    num_iterations: number of iterations to run
    lambda_:        learning rate
@OUTPUT:
    new_fd:         FidesDataset with approximated data
"""
def sampling_based_inference(fd, epsilon, cramers_v = 0.2):
    #TODO: Set this to return nothing if there are non-categorical columns
    #Get the data to be approximated
    data = fd.data_to_use[fd.cat_cols] #Can only approximate categorical variables!!
    #Run the sampling based inference procedure
    print("Initialize dep graph")
    G = initialize_dependecy_graph(epsilon/2, data, cramers_v)
    print("Get j tree")
    jt = get_junction_tree(G)
    print("Calculate marginals")
    marginals = build_marginals(jt, data, epsilon/2)
    print("get approximate data")
    approx_df = generate_synthetic_from_marginals(G.nodes(), jt, marginals, data.shape[0])
    #Reorder the new data for consistency
    approx_df = approx_df[fd.cat_cols]
    method_summary = "Sampling-Based Inference: " + str(epsilon) + " " + str(cramers_v)
    new_fd = FidesDataset(method_summary, fd, approx_df)
    return new_fd
#TODO: This can break if the chosen combination of attributes from another cluster had a 0-count in this cluster due to noise.
#So we need some sort of catch that says "if you haven't seen this, treat it as uniform" or something similar


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~ PrivBayes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
HELPER METHOD
Generate the bayesian network
@INPUT:
    df: data to build BN from
    k   degree of the bayesian network
@OUTPUT:
    bn: bayesian network
"""
def build_bayesian_network(df, k, epsilon):
    bn = []
    V = tuple()
    attributes = list(df.columns)
    x = np.random.choice(attributes)
    bn.append((x, list()))
    V += (x,)
    i = 1
    attributes.remove(x)
    n,d = df.shape
    S = (2/n)*np.log2((n+1)/2)+(n-1)/n*np.log2((n+1)/(n-2))
    delta = (d-1)*S/epsilon
    while attributes:
        if i <= k:
            parent_combos = [V]
            i += 1
        else:
            parent_combos = combinations(V, k)
        omega = dict()
        for x in attributes:
            for combo in parent_combos:
                parent_values = ['\t'.join(map(str, tuple(row))) for index, row in df[[x for x in combo]].iterrows()]
                mi_score = mutual_info_score(parent_values, df[x])
                omega[(x, combo)] = mi_score
        omega_private = {key:np.exp(omega[key]/(2*delta)) for key in omega}
        total_omega_private = sum(omega_private.values())
        omega_private = {key:value/total_omega_private for key,value in omega_private.items()}
        keys = list(omega_private.keys())
        max_pair_index = np.random.choice(list(range(len(keys))), p=list(omega_private.values()))
        max_pair = keys[max_pair_index]
        #max_pair = max(omega, key=omega.get) #NOTE: NOT PRIVATE RIGHT NOW
        print(max_pair)
        bn.append(max_pair)
        V += (max_pair[0],)
        #Need to remove whatever we added as we go
        attributes.remove(max_pair[0])
    return bn

"""
HELPER METHOD
Generate the noisy conditionals
@INPUT:
    df:                 data to build conditionals from
    bn:                 bayesian network of df
    k:                  degree of the bayesian network
    epsilon:            privacy budget for this step
@OUTPUT:
    noisy_conditionals: list of noisy conditionals in x order
"""
def generate_noisy_conditionals(df, bn, k, epsilon):
    (n,d) = df.shape
    noisy_conditionals = [dict() for i in range(d)]
    front_marginal = []
    for i in range(k, d):
        (x, pi) = bn[i]
        attributes = [x]
        for attr in pi:
            attributes.append(attr)
        marginal = (df.groupby(attributes).size()/n).reset_index(name='counts')
        marginal['counts'] += np.random.laplace(loc = 0,scale = (2*(d-k))/(n*epsilon),size=marginal.shape[0])
        counts = marginal._get_numeric_data()
        counts[counts < 0] = 0
        total_count = counts.sum()
        counts = counts/total_count
        marginal['counts'] = counts
        if i == k:
            front_marginal = marginal
        #now that we have the marginal, we need to re-frame it as a conditional over the parent values
        marginal_parent_sums = marginal.groupby(pi)['counts'].agg('sum').reset_index()
        marginal_sum_dict = dict()
        for index, row in marginal_parent_sums.iterrows():
            parent_vals = tuple(row[attr] for attr in pi)
            marginal_sum_dict[parent_vals] = row['counts']
        conditional = dict()
        parents_with_zero_sums = set()
        for index, row in marginal.iterrows():
            parent_vals = tuple(row[attr] for attr in pi)
            marginal_prob = row['counts']
            marginal_sum = marginal_sum_dict[parent_vals]
            if not parent_vals in conditional:
                conditional[parent_vals] = dict()
            if marginal_sum == 0:
                parents_with_zero_sums.add(parent_vals)
                conditional[parent_vals][row[x]] = -1 #Something funky needs to happen if we've set the entire marginal to 0 - I guess everything just gets a random proportion, but how to do this properly?
            else:
                conditional[parent_vals][row[x]] = marginal_prob/marginal_sum
        for parent_set in parents_with_zero_sums:
            num_entries = len(conditional[parent_set])
            for key in conditional[parent_set]:
                conditional[parent_set][key] = 1/num_entries
        noisy_conditionals[i] = conditional
    for j in range(k-1):
        i = k-j-1
        (x, pi) = bn[i]
        attributes = tuple(pi + (x,))
        front_marginal = front_marginal.groupby(attributes)['counts'].agg('sum').reset_index()
        marginal_parent_sums = front_marginal.groupby(pi)['counts'].agg('sum').reset_index()
        marginal_sum_dict = dict()
        for index, row in marginal_parent_sums.iterrows():
            parent_vals = tuple(row[attr] for attr in pi)
            marginal_sum_dict[parent_vals] = row['counts']
        conditional = dict()
        parents_with_zero_sums = set()
        for index, row in front_marginal.iterrows():
            parent_vals = tuple(row[attr] for attr in pi)
            marginal_prob = row['counts']
            marginal_sum = marginal_sum_dict[parent_vals]
            if not parent_vals in conditional:
                conditional[parent_vals] = dict()
            if marginal_sum == 0:
                parents_with_zero_sums.add(parent_vals)
                conditional[parent_vals][row[x]] = -1 #Something funky needs to happen if we've set the entire marginal to 0 - I guess everything just gets a random proportion, but how to do this properly?
            else:
                conditional[parent_vals][row[x]] = marginal_prob/marginal_sum
        for parent_set in parents_with_zero_sums:
            num_entries = len(conditional[parent_set])
            for key in conditional[parent_set]:
                conditional[parent_set][key] = 1/num_entries
        noisy_conditionals[i] = conditional
    #now just group the last thing by the last value
    x = bn[0][0]
    conditional = dict()
    marginal_sums = front_marginal.groupby(x)['counts'].agg('sum').reset_index()
    for index, row in marginal_sums.iterrows():
        conditional[row[x]] = row['counts']
    noisy_conditionals[0] = conditional
    return noisy_conditionals

"""
HELPER METHOD
Sample a synthetic dataset from the noisy conditionals
@INPUT:
    n:              number of rows to be sampled
    bn:             bayesian network
    conditionals:   noisy conditionals
@OUTPUT:
    df:             fully synthetic dataframe
"""
def sample_data(n, bn, conditionals):
    #first just sample the first attribute
    first_column = np.random.choice(list(conditionals[0].keys()), p=list(conditionals[0].values()), size=n)
    df = pd.DataFrame({bn[0][0]:first_column})
    for i in range(1, len(conditionals)):
        (x,pi) = bn[i]
        conditional = conditionals[i]
        new_column = []
        for index, row in df.iterrows():
            parent_values = tuple(row[attr] for attr in pi)
            try:
                conditional_to_choose = conditional[parent_values]
            except KeyError as e: #There's a chance that one marginal sampled something another didn't.
                x_vals = set()
                for cond in conditional.values():
                    for value in cond.keys():
                        x_vals.add(value)
                #x_vals = set(value for value in cond.keys() for cond in conditional.values())
                conditional_to_choose = {x_val:1/len(x_vals) for x_val in x_vals}
            new_column.append(np.random.choice(list(conditional_to_choose.keys()), p=list(conditional_to_choose.values())))
        new_column_df = pd.DataFrame({x:new_column})
        df = pd.concat([df,new_column_df], axis=1)
    return df

"""
Run the PrivBayes code and return a new FidesDatset
NOTE! This can only run over categorical variables.
@INPUT:
    fd:             FidesDatset being approximated
    epsilon:        privacy budget
    k:              dimension of approximation
@OUTPUT:
    new_fd:         FidesDataset with approximated data
"""
def privbayes(fd, epsilon, k):
    df = fd.data_to_use[fd.cat_cols]
    bn = build_bayesian_network(df, k, epsilon/2)
    conditionals = generate_noisy_conditionals(df, bn, k, epsilon/2)
    approx_df = sample_data(df.shape[0], bn, conditionals)
    method_summary = "DPPro: " + str(epsilon) + " " + str(k)
    new_fd = FidesDataset(method_summary, fd, approx_df)
    return new_fd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~ DPPro ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
HELPER METHOD
get the projection matrix from d to k
@INPUT:
    d: number of columns in original data
    k: dimension of approximation
@OUTPUT:
    R: (d x k) projection matrix
"""
def get_projection(d, k):
    R = np.array(np.random.normal(0,1/k,size=(d,k)))
    return R

"""
HELPER METHOD
get the (n x k) noise matrix
@INPUT:
    n:      number of rows in original data
    k:      dimension of approximation
    sigma:  noise scale term
@OUTPUT:
    phi:    (n x k) noise matrix
"""
def get_noise_matrix(n, k, sigma):
    phi = np.array(np.random.normal(0,sigma**2,size=(n,k)))
    return phi

"""
Run the DPPro code and return a new FidesDatset
@INPUT:
    fd:             FidesDatset being approximated
    epsilon:        privacy budget
    delta:          parameter of epsilon-delta differential privacy
    k:              dimension of approximation
@OUTPUT:
    new_fd:         FidesDataset with approximated data
"""
def dppro(fd, epsilon, delta, k):
    X, num_parameters, pivoted_columns = normalize_data(fd)
    #Define important parameters
    (n,d) = X.shape
    upper_delta = max(1 / np.exp(epsilon), 1/2)
    if delta > upper_delta:
        print("Delta value of", delta, "out of bounds, adjusting to", upper_delta)
        delta = upper_delta
    lower_k = int(np.ceil(2*(np.log2(d) + np.log2(2/delta))))
    if k < lower_k or k > d:
        if k > d:
            new_k = d
        else:
            new_k = min(lower_k, d)
        print("K value of ", k, " out of bounds, adjusting to ", new_k)
        k = new_k
    sigma = 4 / epsilon * np.sqrt(np.log2(1/delta))
    print("n", n)
    print("d", d)
    print("k", k)
    print("epsilon", epsilon)
    print("delta", delta)
    print("sigma", sigma)
    #Get the projection matrix R by sampling from N(0, 1/k)
    R = get_projection(d, k)
    #Compute reduced form and add noise
    Y = np.dot(X,R)
    phi = get_noise_matrix(n,k,sigma)
    P = Y + phi
    #Get approximate form by multiplying by pseudo-inverse of R
    approx_X = np.dot(P, np.linalg.pinv(R))
    #Use the approximation to get an approximated DataFrame
    approx_df = reconstruct_data(fd, approx_X, num_parameters, pivoted_columns)
    method_summary = "DPPro: " + str(epsilon) + " " + str(k)
    new_fd = FidesDataset(method_summary, fd, approx_df)
    return new_fd

if __name__ == "__main__":
    df = pd.read_csv("data/adult.csv")
    df = df[["income","workclass","sex","education","native-country","relationship"]]
    epsilon = 1
    bn = build_bayesian_network(df, 3, epsilon/2)
    conditionals = generate_noisy_conditionals(df, bn, 3, epsilon/2)
    sampled_df = sample_data(df.shape[0], bn, conditionals)
    print(sampled_df)
