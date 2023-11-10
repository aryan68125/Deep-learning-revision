import numpy as numpy
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
   print(tabulate(df, headers='keys', tablefmt='psql'))

   '''
   In this dataframe ecommerce_data.csv the user_action column is our target and every other
   columns in this dataframe is our input or features that we are gonna feed into our ML model
   '''
   
   #displaying histogram of different columns in the dataframe
   '''
   The histogram shows the distribution of values in the column time_of_day in the dataframe
   '''
   df['time_of_day'].plot(kind='hist')
   plot.pyplot.show()

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
get_data()