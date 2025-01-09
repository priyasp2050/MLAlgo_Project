#1)Univariate Feature Selection
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
#from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
filename = 'indians-diabetes.data.csv'
hnames = ['preg', 'plas', 'pres', 'skin', 'test','mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=hnames)
array = dataframe.values
X = array[ : , 0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)  #how x related with y
# summarize scores
#set_printoptions(precision=3)
print( fit.scores_ )

#[ 111.52 1411.887 17.605 53.108 2175.565 127.669 5.393 181.304]
# 1 2 3 4 5 6 7 8

# 'preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age'
#            2                       5      6               8
features = fit.transform(X)
# summarize selected features
print( "\n\n" )
print(features[0:20,:])