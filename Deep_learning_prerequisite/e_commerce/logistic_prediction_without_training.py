import numpy as np
import pandas as pd
from pre_processing import get_binary_data

'''
this file demostrates how a logistic regression makes predictions 
training of logistic regression demo is excluded here for the time being. 
here we will only focus on how to make predictions at the moment without worring about 
training the Logistic regression model
'''

#hence we aill just load in the train set and ignore the test set
X, Y, _, _ = get_binary_data()

'''
Randomly initialize the logistic regression model weights
in order to know the size of the weights we need to know the dimensionality of X or the
number of columns in X
X = N,D --> N = number of samples (number of rows) , D = number of Features (number of columns)
D = X.shape[1] here we only want D the shape value at index 1 of X
'''
D = X.shape[1]

'''
From here we can initialize the weights.
weight is a one dimensional vector of size D
we will initialize the weights randomely from the standard normal distribution
'''
W = np.random.randn(D)

'''
b is just a scalar bias term
'''
b = 0

#Make predictions : 
def sigmoid(a):
    return 1/(1+np.exp(-a))

'''
define a forward function , which takes in an input and computes the output
so in other words it's computing sigmoid(W^t.X)
here dot = matrix multiplication
X = shape (N,D) and W = shape D matrices hence X matrix when multiplied with W matrix makes
sence since it satisfies the matrix multiplication law.
matrix multiplication law states that we are allowed to multiply two matrices when the 
inner dimensions of two matrices matches with each other.
'''
def forward(X,W,b):
    return sigmoid(X.dot(W) + b)

'''
calculating the probability of Y whne x is given
'''

P_Y_given_X = forward(X, W, b)

'''
p_y_given_X is a one dimensional scalar value since multiplication of two vectors is a scalar
'''
print(f"P_Y_given_X values = {P_Y_given_X}")
print(f"p_y_given_X shape = {P_Y_given_X.shape}")