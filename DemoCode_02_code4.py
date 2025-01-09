#4) Feature Selection based on IG Importance Index
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

filename = 'indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test','mass', 'pedi', 'age', 'class' ]
#         0         1        2         3        4       5           6       7          8
dataframe = pd.read_csv(filename, names=names)

array = dataframe.values

X = array[:,0:8]

Y = array[:, 8]

model = ExtraTreesClassifier() #

model.fit(X, Y)

scores = model.feature_importances_

print("scores = ", scores )

result = list(zip(names , scores))
# 0 , 1
print("\n\n", result )

from operator import  itemgetter
print("\n\n After sorting = \n", sorted(result, key=itemgetter(1) , reverse= True))