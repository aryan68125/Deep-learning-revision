import numpy as np
import matplotlib.pyplot as plt
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
y = sigmoid(z)
print (y)

# Plotting
'''
plt.scatter: This function creates a scatter plot.
x[:,0] and x[:,1]: These are the x and y coordinates of the data points. 
x[:,0] represents all rows of the first column of the x array, and x[:,1] represents all rows of the second column of the x array. 
This is because x is a 2D array with shape (N, D), where N is the number of samples and D is the number of features.
c=y: This parameter specifies the color of each data point. Here, y represents the probabilities calculated by the logistic regression model. 
It assigns colors to the data points based on these probabilities.
cmap='bwr': This parameter specifies the colormap to be used for coloring the data points. Here, 'bwr' stands for blue-white-red colormap, 
which maps low values to blue, medium values to white, and high values to red.
plt.xlabel: This function sets the label for the x-axis of the plot. Here, it sets the label to 'X1'.
plt.ylabel: This function sets the label for the y-axis of the plot. Here, it sets the label to 'X2'.
plt.title: This function sets the title of the plot. Here, it sets the title to 'Logistic Regression Result'.
plt.colorbar: This function adds a colorbar to the plot, which shows the mapping between colors and values. 
Here, it adds a colorbar and sets its label to 'Probability', 
indicating that the colors represent the probabilities calculated by the logistic regression model.
plt.show: This function displays the plot on the screen. It should be called after setting up all the elements of the plot.
'''
plt.scatter(x[:,0], x[:,1], c=y, cmap='RdYlGn_r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression Result')
plt.colorbar(label='Probability')
plt.show()