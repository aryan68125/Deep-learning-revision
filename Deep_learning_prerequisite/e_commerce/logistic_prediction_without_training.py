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

'''
So here what we have created is a binary classifier so we want our predictions to be 0 or 1
In order to do that we have to round these probabilities that we are getting from 
the forward function. so if the probability is greater than 50% then we will say it's a 1
otherwise we aill say it's a zero.
'''
predictions = np.round(P_Y_given_X)
print(f"predictions = {predictions}")

'''
Let's check the classification rate of our model. 
The function below will check the classification rate for our model
The function takes in targets (Y) and Predictions
np.mean(Y==P)
Y == P it returns true if Y = P and if Y!= P then it returns false since we are 
writing this inside a numpy array it will check the equality for each element in the array
and return a numpy array with truth and false values in it.
In python the values of true and false are treated as zeros and ones.
So the classification rate if we had an array of zeros and ones, where one means the 
prediction was right and zero means the prediction was not right. 
the classification rate will be the number of ones divided by the total number of samples.
'''
def classification_rate(Y,P):
    return np.mean(Y==P)

classification_rate_values = classification_rate(Y,predictions)
print(f"score = {classification_rate_values}")