import numpy as np
N = 100 # number of rows or samples
D = 2 # number of columns or features

# generating random data points for us to use
# X = data that we need to perform logistic regression on
# X = normally distributed data matrix
X = np.random.randn(N,D)

'''
So this time around we are going to have labels right because we
need to calculate the error.

I want my first 50 points to be centered at x = -2 and y = -2
So we can do that by matix of ones and multiplying it by -2
X[:50 , :] = X[:50,:] - 2*np.ones((50,D)) --> class 1

I want class 2 to be centered around the point x = 2 and y = 2
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) --> class 2
'''
X[:50 , :] = X[:50,:] - 2*np.ones((50,D)) #class 1
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) #class 2

'''
creating an array of Targets 
Targets = actual output in the synthesized dataframe
the first 50 I am going to set to 0 and the last 50 I am going to 
set to 1
T = np.array([0]*50 + [1]*50)
'''
T = np.array([0]*50 + [1]*50) # targets

'''
Here we are gonna concatenate the column of ones
concatenate a column with 1s or biase
Now we know that we are gonna have to add a bias term. 
In order to do that we just gonna add the column of ones to the original data and include
the bias term in the weights w.
An array in numpy is only one Dimensional and we need it to be 2 dimensional in order 
to have n rows and two column
So now I am gonna concatenate the array of ones to my original dataset by 
"Xb = np.concatenate((ones,X),axis=1)"

NOTE : If you write ones = np.array([1]*N).T --> Then it will generate the error
numpy.AxisError: axis 1 is out of bounds for array of dimension 1
What you need to actually do is --> ones = np.array([[1]*N]).T -->then this error will be
resolved
'''
ones = np.array([[1]*N]).T
print(f"ones shape = {ones.shape}")
print(f"X shape = {X.shape}")
Xb = np.concatenate((ones,X),axis=1)

'''
From here we can initialize the weights.
weight is a one dimensional vector of size D+1
we will initialize the weights randomely from the standard normal distribution
'''
W = np.random.randn(D+1)

'''
Calculating the dot product of matrix Xb (input matrix) and W (Weights)
'''
Z = Xb.dot(W)

'''
Feeding the Z (dot product of matrix Xb (input matrix) and W (Weights)) in the 
Sigmoid function
# NOTE : Numpy works on vectors as well as scalers so you need to pass it through the 
         function once here you will get the values will be in between 0 and 1 and 
         our output is n by 1 matrix
'''
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

'''
Calculating the output Y From the Logistic regression model
Y = calculated output via logistic regression model 
'''
Y = sigmoid(Z)

'''
Calculating the cross entropy error
This cross_entropy_error function takes in Targets T and Predicted Output Y 
and It's just sums over each individual cross entropy here for each sample
'''
def cross_entropy_error(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E-=np.log(Y[i])
        else:
            E-=np.log(1-Y[i])
    return E



'''
learning rate is the step size that we take in the direction of the derivative so that
we reach at the bottom of the gradient descent curve where the derivative is zero
when the equation's or the objective function's (Cost function = error function = objective function)
derivative reaches to zero the weights stops updating
'''
learning_rate=0.1
'''
I am goint to do 100 iterations of gradient descent
'''
for i in range(100):
    '''
    printing out the cross entropy every 10 steps so that we can see that it's decreasing
    '''
    if (i%10 == 0):
        print(f"cross entropy error = {cross_entropy_error(T,Y)}")
    
    '''
    Here we are finding weights using gradient descent method 
    '''
    W+= learning_rate * np.dot((T-Y).T, Xb)
    '''
    Recalculate the output Y (the predictions of logistic regression)
    '''
    Y = sigmoid(Xb.dot(W))

'''
printing out the final weights for this logistic regression model after the training 
of this model is complete
'''
print(f"weights after the training of logistic regression model : {W}")