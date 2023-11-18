import numpy as np
N = 100 # number of rows or samples
D = 2 # number of columns or features

# generating random data points for us to use
X = np.random.randn(N,D)

'''
So this time around we are going to have labels right because we
need to calculate the error.

I want my first 50 points to be centered at x = -2 and y = -2
So we can do that by matix of ones and multiplying it by -2
X[:50 , :] = X[:50,:] - 2*np.ones((50,D)) --> class 1

I want class 2 to be centered around the point x = 2 and y = 2
X[50:,:] = X[50:,:] - 2*np.ones((50,D)) --> class 2
'''
X[:50 , :] = X[:50,:] - 2*np.ones((50,D)) #class 1
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) #class 2

'''
creating an array of Targets 
the first 50 I am going to set to 0 and the last 50 I am going to 
set to 1
T = np.array([0]*50 + [1]*50)
'''
T = np.array([0]*50 + [1]*50) # targets

ones = np.array([1]*N).T
Xb = np.concatenate((ones, X), axis =1)