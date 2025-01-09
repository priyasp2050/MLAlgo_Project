# Data Transformation:
#-----------------------
#Prepare the Data for AI/ML Algorithm
#***************
from array import array

import numpy as np
# 4 ways to prepare the Data for AI/ML Algorithm
#------------------------------------------------
# 1. Rescale data
# 2. Standardize data
# 3. Normalize data
# 4. Binarize data


#Stpes of Data Transforms
#-------------------------
#Steps-1 : Load the dataset from a URL.

#Step-2 : Split the dataset into the input and output variable for ML

#Step-3 : Apply a pre-processing transformation technique to transform only the input variables.

#Step-4 : Summarize the data to show the change.

#Rescale data (custom range between 1 to 10)
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

filename = 'indians-diabetes.data.csv'
hnames=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=hnames)
array = dataframe.values
#separate array into input and output components
X = array[ :, 0:8] #[row, cols]
Y = array[: , 8]
scaler = MinMaxScaler(feature_range=(1,10)) #Range

#First Method
rescaledX = scaler.fit_transform(X)

#Second Method
scaler = scaler.fit(X)
rescaledX = scaler.transform(X)

print( rescaledX[0:30 , : ])

print("\n\nMean of First coloumn=" , end="")
print(np.mean( rescaledX[ : , 0]))

print("\n\nMean of Second coloumn=" , end="")
print(np.mean( rescaledX[ : , 1]))

