#Binarize Data (Make Binary)
#-----------------------------
import numpy as np
from sklearn.preprocessing import Binarizer

X = np.array( [ [ 1., -1., 2.],
[ 2., 0., 0.],
[ 0., 1., -1.] ] )

#Default Threshold value is 0
transformer = Binarizer(threshold=1)
data2 = transformer.transform(X)

print( data2 )
