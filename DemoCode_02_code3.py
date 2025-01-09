#PCA: Dimension Reduction Method
#-------------------------------------
# Feature Extraction with PCA :
# PCA is a dimension reduction technique

import pandas as pd
from sklearn.decomposition import PCA
filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test','mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

pca = PCA(n_components=3) # feature extraction via PCA
fit = pca.fit(X)
resultX = pca.transform(X)
print( "\nResult : \n" , resultX )

# summarize components
print( "Explained Variance:" ,fit.explained_variance_ratio_ )