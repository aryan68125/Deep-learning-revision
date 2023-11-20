'''
In this file we are gonna look at how to train logistic regression model
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pre_processing import get_binary_data

# get binary data using pre_processing module's get_binary_data function
Xtrain, Ytrain, Xtest, Ytest = get_binary_data()

#randomly initialize the weights
'''
From here we can initialize the weights.
weight is a one dimensional vector of size D+1
we will initialize the weights randomely from the standard normal distribution

Remember : 
X is of shape N by D and we want D whichs is the second dimension of the shape tuple.
D = Xtrain.shape[1]

Then we are goind to create our randomly initialized our weight vector.
'''
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0

#making predictions
'''
Feeding the a (dot product of matrix Xb (input matrix) and W (Weights)) in the 
Sigmoid function. a = activation function
# NOTE : Numpy works on vectors as well as scalers so you need to pass it through the 
         function once here you will get the values will be in between 0 and 1 and 
         our output is n by 1 matrix
'''
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

#measure the performance of the logistic regression model
'''
So previously we had a function to compute the classification rate.
Now we also have the cross entropy loss So there two different metrics to measure
the performance of this classifier.
'''
#method 1 --> classification rate method to measure the performance of the logistic regression model
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

In order to get the output from classification_rate function which tells the accuracy of the logistic regression model
all you need to do is call the function as shown below
classification_rate(Ytest,np.round(pYtest))}
NOTICE how wehave rounded the predictions done by the logistic regression model(pYtest).
'''
def classification_rate(Y,P):
    return np.mean(Y==P)

# method 2 --> cross entropy loss function to measure the performance of the model
'''
Calculating the cross entropy error
This cross_entropy_error function takes in Targets T and Predicted Output Y 
and It's just sums over each individual cross entropy here for each sample

This is the formula for the cross entropy loss 
negatice of the sample mean -np.mean()
-np.mean() is a vectorize computation so it's doing all of the samples at once
So the complete equation becomes 
-np.mean(Y * np.log(pY) + (1 - Y)*np.log(1 - pY))
Y = Targets
pY = predicted output of Logistic regression model 
'''
def cross_entropy_error(Y, pY): # pY is the p of y given x P(Y|X)
    return -np.mean(Y * np.log(pY) + (1 - Y)*np.log(1 - pY))

# So now here we have a training loop
'''
So we will begin by initializing a couple of list to store loss per iteration.
Remember cost is just a synonym of loss
'''
train_costs = [] # It accumulates the cross entropy error during trainig for the train dataset
test_costs = [] # It accumulates the cross entropy error during training for the test dataset
accuracy_costs_train = [] # It accumulates the classification rate of the model during training
accuracy_costs_test = []
'''
learning rate is the step size that we take in the direction of the derivative so that
we reach at the bottom of the gradient descent curve where the derivative is zero
when the equation's or the objective function's (Cost function = error function = objective function)
derivative reaches to zero the weights stops updating
'''
learning_rate = 0.001

for i in range(30000):
    pYtrain = forward(Xtrain,W,b)
    pYtest = forward(Xtest,W,b)
    ctrain = cross_entropy_error(Ytrain, pYtrain)
    ctest = cross_entropy_error(Ytest, pYtest)
    classification_accuracy_train = classification_rate(Ytrain, np.round(pYtrain))
    classification_accuracy_test = classification_rate(Ytest, np.round(pYtest))
    accuracy_costs_train.append(classification_accuracy_train) 
    accuracy_costs_test.append(classification_accuracy_test)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    #now the next step is to do gradient descent 
    '''
    Gradient desecnt is a method through which we are gonna train our logistic regression model
    '''
    '''
    Here we are finding weights using gradient descent method 

    NOTE : 
    earlier we did gradient descent using this formula ```    W+= learning_rate * np.dot((T-Y).T, Xb) ```
    now we are doing gradient descent using this ``` W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain) ```
    Ytrain = Targets 'T'
    pYtrain = Predictions done by logistic regression model 'Y'
    Xtrain = input matrix 'X'
    Xtrain.T means Xtrain Transpose i.e transposing the xtrain matrix
    '''
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain)

    '''
    Now we are gonna update the bias
    for updating the bias we do learning_rate * gradient
    gradient = (pYtrain - Ytrain).sum()
    '''
    b -= learning_rate * (pYtrain - Ytrain).sum()

    #printing out the losses every few steps
    if (i%10 == 0):
         print(f"epochs = {i}, cross entropy train : {ctrain}, cross entropy test = {ctest}, classification rate (train)= {classification_rate(Ytrain,np.round(pYtrain))} classification rate (test) = {classification_rate(Ytrain,np.round(pYtrain))}")

#print the final classification rate for the logistic regression model
'''
Remeber we accumulated the cross entropy errors for train and test data in 
train_costs = []
test_costs = []
'''
print(f"final accuracy of the logistic regression model : {classification_rate(Ytest,np.round(pYtest))}")

'''
The next step is to get the sense of wheather or not we chose the correct learning rate or the correct number of 
epochs. So we are gonna look at train costs and test costs per epoch
'''
plt.plot(train_costs, label='train costs')
plt.plot(test_costs, label='test cost')
plt.plot(accuracy_costs_train, label='classification_accuracy train')
plt.plot(accuracy_costs_test, label='classification_accuracy test')
plt.legend()
plt.show()