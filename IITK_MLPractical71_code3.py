# Normalize Data (Example-1)
#---------------------
#Say your features were x, y and z Cartesian co-ordinates, your
# scaled value for x would be:

#X_scaled = Xi / sqrt( Xi**2 + Yi**2 + Zi**2 )

#Each point is now within 1 unit of the origin on this Cartesian co-ordinate system.

#Normalizer works on the rows, not the columns#X_scaled = Xi / sqrt( Xi**2 + Yi**2 + Zi**2 )#NOTE: i represents row index#X_scaled = X / sqrt( x^2 + y^2 + z^2 )import numpy as np
from sklearn.preprocessing import Normalizer
import numpy as np
X = np.array( [ [4, 1, 2, 2],
[1, 3, 9, 3],
[5, 7, 5, 1] ] )

transformer = Normalizer().fit(X) # fit does nothing.

print( transformer.transform(X) )