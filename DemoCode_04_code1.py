# 1)-Evaluate using a Train and Test Set split
import warnings
warnings.filterwarnings(action="ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

filename = 'indians-diabetes.data.csv'
hnames = ['preg', 'plas', 'pres','skin',
'test','mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=hnames)
array = dataframe.values
X = array[: ,0:8]
Y = array[: , 8]
test_data_size = 0.33
seed = 7
#random_state = seed
#to regenerate random number we use seed value

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_data_size ,  random_state=seed)
#  1      2         3         4

print("67% =len(X_train) = ", len(X_train) )
print("33% =len(X_test) = ", len(X_test) )
print("67% =len(Y_train) = ", len(Y_train) )
print("33% =len(Y_test) = ", len(Y_test) )  #254

model = LogisticRegression()

model.fit(X_train, Y_train) # 67% of historical Data

result = model.score(X_test, Y_test) #33% of testing data

print( "Accuracy= %.3f%%" % (result * 100) )