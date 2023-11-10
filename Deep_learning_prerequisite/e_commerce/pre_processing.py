import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib as plot
import os

'''
get data function
'''
def get_data():
   # load our data by reading the csv file in the file manager
   df = pd.read_csv('ecommerce_data.csv')
   #read the first few entries in the dataframe after reading the csv file
   '''df.head() will only work in a jupyter or google colab notebook.
    It won't show output in your terminal'''
#    print(tabulate(df, headers='keys', tablefmt='psql'))

   '''
   In this dataframe ecommerce_data.csv the user_action column is our target and every other
   columns in this dataframe is our input or features that we are gonna feed into our ML model
   '''
   
   #displaying histogram of different columns in the dataframe
   '''
   The histogram shows the distribution of values in the column time_of_day in the dataframe
   '''
#    df['time_of_day'].plot(kind='hist')
#    plot.pyplot.show()

   '''
   It's much more easier to work with numpy arrays. hence we have to convert this 
   dataframe into a numpy array.
   data = df.to_numpy() right now as of 2023 This function converts a dataframe into 
   a numpy array so in the older code you migh see that in order to convert a dataframe 
   there people used to write 'df.as_matrix()' or 'df.values()' it all means the same
   i.e these all are converting a dataframe into a numpy array.
   '''
   '''
   Points to remember : 
   avoid using np.asmatrix(df) if df.as_matrix() is not working these functions may look the same 
   but reality they are not the same. The reason they are not the same is because 
   df.as_matrix() --> does not return an actual matrix object it returns a numpy array
   np.asmatrix(df) --> it returns an actual matrix
   '''
   data = df.to_numpy()

   #shuffle the data 
   '''
   This function ``` np.random.shuffle(data) ``` is a little wierd because it doesn't return
   anything becasue it shuffle's the data that you pass in, in it's place by the row.
   NOTE : The columns don't get shuffled it only shuffles's the rows.
   '''
   np.random.shuffle(data)

   '''
   Now we are gonna split the features(inputs denoted by x1, x2, ...) and the 
   labels(desired output/targets denoted by y).
   x = data[:,:-1] means that we are selecting all the rows and all the columns up to second
   last column hence -1
   x = data[rows,columns] --> x = data[:,:-1]
   '''
   x = data[:,:-1]
   '''
   y = data[:,-1] selecting targets , select all the rows and only the last column from the numpy array
   and also convert it into an int datatype
   '''
   y = data[:,-1].astype(np.int32)

   '''
   The time_of_day column of the datatype is catogorical in nature i.e there are different
   categories of time during which the users have either logged_in or logged out. 
   the categores in this time_of_day column of the ecommerce.csv file is 0, 1, 2, 3
   since categorical data is encoded as integers it's not appropriate for 
   machine learning model 
   The simple solution to this problem is to ONE HOT ENCODE the categorical columns
   present in the dataframe 
   ONE HOT ENCODING is usually performed on the input data (It's the x in our case here)
   N,D = x.shape --> N is the number of rows in the numpy array and D is the number of columns
   in the numpy array
   x2 = np.zeros(N,D+3) --> Why D+3 ? the reason for this is that when performing ONE HOT
   ENCODING each category get's it's own new column in the dataframe and those rows who 
   belonged to a particular category gets 1 as a value in that particular category column.
   since we can also replace the existing column we only need 3 more column to create total
   of 4 new columns for the respective time_of_day column in the dataframe.
   so the time_of_day column in the dataframes gets replaced by each of the catagories
   present in the time_of_day column and each category gets a new column of it's
   own in the dataframe
   --:-> here we are adding 3 new columns in x2 numpy array matrix as compared to the old
   numpy array matrix x to accomodate categories columns after ONE HOT ENCODING is 
   complete
   '''
   N,D = x.shape
   x2 = np.zeros((N,D+3))
   #copy the non categorical data in the new x2 numpy array from the old x numpy array
   '''
   x2[:,:(D-1)] --> all rows , columns up to D-1 as we know the last column is the
   time_of_day column which is categorical in nature
   x2[:,:(D-1)] = x[:,(D-1)] --> hence all the non categorical type columns are 
   being copied here from the old numpy array to a new one
   '''
#    x2[:,:(D-1)] = x[:,(D-1)] -->not working
   x2[:,0:(D-1)] = x[:,0:(D-1)]
   #now the actual one hot encoding will take place here
   '''
   Now perfoming ONE HOT ENCODING on the columns that are categorical in nature
   z = np.zeros((N,4)) --> here we are creating a new numpy array matrix named z
   this matrix z have N rows and 4 columns which is what we need since the number of 
   categories in time_of_day column of the ecommerce dataframe has 4 categories
   z = [np.arange(N), x[:,(D-1)].astype(int32)] = 1 --> 
   so bassically here we are passing rows and columns in the new numpy matrix z
   z [(row1, ro2 , row3...) , (col1, col2, col3...)] = value
   Now all we have to do is set the last 4 columns of the new numpy matrix x2 to z
   numpy matrix
   x2[:,-4:] = z --> we are selecting the last 4 columns from the x2 numpy matrix 
   and setting to the z matrix hence all the values of each columns will be 
   copied to x2 matrix from the z matrix
   '''
   z = np.zeros((N,4))
   z[np.arange(N), x[:,(D-1)].astype('int32')] = 1
   x2[:,-4:] = z

   '''
   now assign the values in x2 back to the original numpy matrix x 
   '''
   x = x2
   # now split the data into a train and test sets
   '''
   xtrain = x[:-100] --> all the rows except the last 100 rows will copied to the xtrain 
   varibale of type numpy array this will be our train set on which our ML model will be trained
   on.
   xtest = x[-100:] --> all the rows starting from -100th row all the way to the end
   will be copied to the xtest of type numpy array this will become our test dataset.
   '''
   xtrain = x[:-100]
   ytrain = y[:-100]
   xtest = x[-100:]
   ytest = y[-100:]
   '''
   since all the columns of our dataframes are numerical we are gonna do a simple
   method of pre-processing which is to standardize our columns also known as 
   normalization
   '''
   for i in range(1,2):
      m = xtrain[:,i].mean() #--> calculating the mean
      s = xtrain[:,i].std() #--> calculating standar deviation
      #standardization/normalization --> give the column 0 mean and unit variance
      xtrain[:,i] = (xtrain[:,i]-m)/s
      xtest[:,i] = (xtest[:,i]-m)/s
   return xtrain , ytrain, xtest, ytest

# x_train, y_train, x_test, y_test = get_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

'''
create a function to convert the data into binary format
in order to do that we are just gonna filter out data that are not zero and not one
x2train = x_train[y_train<=1] --> explaing this code from inside towards outside
y_train<=1 --> this compares y_train which is n length 1D array since it's shape is (400,)
to some number here in our case it is 1. So this will return a boolean array 
x_train[boolean_values] --> so the boolean values returned by y_train<=1 is converted into 1's
and 0's and it acts as an index for the x_train numpy array. It's gonna keep the value where
the boolean value is true and throw away the value where the boolean value is false.
'''
def get_binary_data():
    x_train, y_train, x_test, y_test = get_data()
    x2train = x_train[y_train<=1]
    y2train = y_train[y_train<=1]

    x2test = x_test[y_test<=1]
    y2test = y_test[y_test<=1]
    return x2train, y2train, x2test, y2test

x2train, y2train, x2test, y2test = get_binary_data()
print(x2train.shape)
print(y2train.shape)
print(x2test.shape)
print(y2test.shape)