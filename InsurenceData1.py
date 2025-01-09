import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv('insurance_data2.csv')

print( df.head() )
 

X = df.iloc[ : , 0:1].values
#            row, col
Y = df.iloc[:, 1].values

print(X)

print(Y)


#display the data using scatter plot
plt.title("Insurance data")
plt.xlabel("Age")
plt.ylabel("Insured or not")
plt.scatter(X,Y)
plt.show()





n = len(X)

m_x = np.mean(X)
m_y = np.mean(Y)

SS_xy = np.sum(X*Y) - n * m_x * m_y
SS_xx = np.sum(X*X) - n * m_x * m_x

m = SS_xy / SS_xx
print("Value of m = ", m)

c = m_y - m * m_x

print("Value of c = ", c)

print("\nModel : y = ", m, " * X + ", c)

y_predicted = m * X + c

print(y_predicted)



plt.title("Insurance data")

plt.xlabel("Age")

plt.ylabel("Insured or not")

plt.scatter(X,Y) # Printing original historical data
plt.plot(X, y_predicted, "ro-") #Print the best file line for the given data
plt.show()






import math

def get_sigmoid(z):
      return 1 / ( 1 + np.power(math.e,-z))


plt.title("Insurance data")

plt.xlabel("Age")

plt.ylabel("Insured or not")

plt.scatter(X,Y,color='b',s=150)

plt.plot(X, get_sigmoid(y_predicted),'ro-')

plt.show()