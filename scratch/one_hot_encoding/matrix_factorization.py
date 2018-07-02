#!/usr/bin/python
#
# Created by Luke Rodriguez (2018)
# Adapted from a script by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import NMF

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 1:
            print("Breaking at step " + str(step))
            break
    return P, Q.T

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    K     : the number of latent features
@OUTPUT:
    the final matrices P and Q
"""
def scikit_als(R,K):
    model = NMF(n_components=K, init='random', random_state=0)
    W = model.fit_transform(R)
    H = model.components_
    return W,H

def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)

"""
Adapted from https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/

@INPUT:
    R     : a matrix to be factorized, dimension N x M
    K     : the number of latent features
    steps : the number of iterations
    lambda_ : the learning rate
@OUTPUT:
    the approximated matrix R_hat

"""
def als_implementation(R,K,steps,lambda_):
    W = R>0.5
    W[W == True] = 1
    W[W == False] = 0
    # To be93444.3197098 consistent with our Q matrix
    W = W.astype(np.float64, copy=False)
    lambda_ = 0.1
    n_factors = K
    m, n = R.shape
#    print(m,n)
    n_iterations = steps
    X = np.random.rand(m, n_factors) 
    Y = np.random.rand(n_factors, n)
    errors = []
    for ii in range(n_iterations):
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), 
                        np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
                        np.dot(X.T, R))
        if ii % 100 == 0:
            print('{}th iteration is completed'.format(ii))
    errors.append(get_error(R, X, Y, W))
    R_hat = np.dot(X, Y)
#    print(R)
#    print(R_hat)
#    print('Error of rated movies: {}'.format(get_error(R, X, Y, W)))
    return R_hat

"""
Adapted from https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/

@INPUT:
    R     : a matrix to be factorized, dimension N x M
    K     : the number of latent features
    steps : the number of iterations
    lambda_ : the learning rate
    epsilon : the privacy budget for the matrix factorization step.
@OUTPUT:
    the approximated matrix R_hat

"""
def als_implementation_with_noise(R,K,steps,lambda_,epsilon=1):
    W = R>0.5
    W[W == True] = 1
    W[W == False] = 0
    # To be consistent with our Q matrix
    W = W.astype(np.float64, copy=False)
    lambda_ = 0.1
    n_factors = K
    m, n = R.shape
#    print(m,n)
    n_iterations = steps
    X = np.random.rand(m, n_factors) 
    Y = np.random.rand(n_factors, n)
    errors = []
    noise = np.random.laplace(0,1/epsilon,size=(n_factors, n))
    for ii in range(n_iterations):
        X = np.linalg.solve(np.dot(Y, Y.T) + np.dot(Y, noise.T) + lambda_ * np.eye(n_factors), 
                        np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
                        np.dot(X.T, R))
        if ii % 100 == 0:
            print('{}th iteration is completed'.format(ii))
    errors.append(get_error(R, X, Y, W))
    R_hat = np.dot(X, Y)
#    print(R)
#    print(R_hat)
#    print('Error of rated movies: {}'.format(get_error(R, X, Y, W)))
    return R_hat
"""
@INPUT:
    R : a matrix to be factorized, dimension N x M
    samples : number of rows in the output matrix
    K : the number of latent features
    steps : the number of iterations
    lambda_ : the learning rate
    epsilon : the privacy budget for the matrix factorization step - if 0 as by default, no noise is added
@OUTPUT:
    A (samples x M) matrix, where M is the number of columns in R
"""
def run_and_sample_als(R,samples,K,steps,lambda_,epsilon=0):
    if epsilon > 0:
        R_hat = als_implementation_with_noise(R,K,steps,lambda_,0.99*epsilon)
    else:
        R_hat = als_implementation(R,K,steps,lambda_)
    
    num_non_zero = round(np.random.laplace(np.sum(R),1/(0.01*epsilon)))
    if num_non_zero >= R.shape[0]: num_non_zero = R.shape[0]-1 #trying to catch weird edge cases, small matrices and small epsilons probably will still cause errors.
    threshold = np.partition(R_hat.flatten(), -num_non_zero)[-num_non_zero]
    output = R_hat>=threshold
    overlap = (np.sum(R)+num_non_zero-np.sum(np.abs(R-output)))/2

    print("Percent overlap: " + str(100*float(overlap)/np.sum(R)))

    #bonus: sample output
    new_df = pd.DataFrame(output[np.random.choice(output.shape[0], samples, replace=True), :])
    return new_df

###############################################################################

