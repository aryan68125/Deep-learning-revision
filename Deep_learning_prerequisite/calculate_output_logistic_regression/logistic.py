import numpy as np
'''
So the first thing is we are gonna need some data in order to do logistic regression on.
'''
N=200
D=2
'''
So I am gonna create a 100 by 2 normally distributed data matrix.
Now we know we are gonna have to add a bias term.
So in order to do that we are gonna add a column of 1's to the original data and include the bias term in the weights w.
'''
x=np.random.randn(N,D)
#an array in numpy is only one dimensional and we need it to be 2 dimensional in order to have n rows and 1 column
ones = np.array([[1]*N]).T
#so now we are gonna concatinate this vector of one's to my original dataset
Xb = np.concatenate((ones,x), axis=1)
#so now I am gonna randomely initialize the weight vector
w = np.random.randn(D+1)

'''
In order to calculate the sigmoid
Step 1 calculate the dot product between each row of x and w
z = Xb.dot(w) --> It does matrix multiplication
here we are using numpy inbuilt function to calculate the multiplication of the two matrices.
'''
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))
print (sigmoid(z))
