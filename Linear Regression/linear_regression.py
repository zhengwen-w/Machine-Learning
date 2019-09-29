"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    w=w.reshape(-1,1)
    SumMatrix=np.dot(X,w)
    n=len(y)
    SumM=np.sum(SumMatrix,axis=1)
    summ=sum(abs(SumM-y))
    err=float(summ)/float(n)
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):

    aa=np.dot(X.T, X)
    y=y.reshape(-1,1)
    aa = np.dot(np.linalg.inv(aa), X.T)
    w=np.dot(aa,y)
    w=np.sum(w,axis=1)
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################	
    return w
	

###### Q1.3 ######
def linear_regression_invertible(X, y):
    aa = np.dot(X.T, X)
    a,b=np.linalg.eig(aa)

    a=abs(a)
    s=min(a)

    while s<10**(-5):
        aa=aa+10**(-1)*np.identity(len(aa))
        a, b = np.linalg.eig(aa)
        a = abs(a)
        s = min(a)

    y = y.reshape(-1, 1)
    aa = np.dot(np.linalg.inv(aa), X.T)
    w = np.dot(aa, y)
    w = np.sum(w, axis=1)
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    aa = np.dot(X.T, X)
    aa=aa+lambd*np.identity(len(aa))
    aa = np.dot(np.linalg.inv(aa), X.T)
    y = y.reshape(-1, 1)
    w = np.dot(aa, y)
    w = np.sum(w, axis=1)
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    minerr=float('inf')
    bestlambda=None
    for j in range(-19,20):
        i=10**j
        w=regularized_linear_regression(Xtrain, ytrain,i)
        w = w.reshape(-1, 1)
        SumMatrix = np.dot(Xval, w)
        n = len(yval)
        SumM = np.sum(SumMatrix, axis=1)
        summ = sum(abs(SumM - yval))
        err = float(summ) / float(n)
        if err<minerr:
            minerr=err
            bestlambda=i
        i=i*10
    
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    return bestlambda

    

###### Q1.6 ######
def mapping_data(X, power):
    aaa=X
    for i in range(2, power+1):
        a=X**i
        aaa=np.concatenate((aaa,a),axis=1)
    
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    
    return aaa


