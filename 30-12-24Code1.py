import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('LinearRegressionPoly_Data.csv')
print(data)
print(data.shape) #(7,2)
X = data.iloc[ : , 0:1].values #[ rows , cols ]
y = data.iloc[ : , 1].values #[ rows ., cols]
print("X.shape = ", X.shape,  "\n X=\n", X)
print("Y.shape = ", y.shape,  "\n Y=\n", y)


#step: Fitting Linear Regression to the datset
#Fitting the Linear Regression model on two components X and y.
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y) #estimate the parameters of the model
y_dash = lin.predict(X)
plt.scatter(X, y, color='blue')
plt.scatter(X, y_dash, color='m')
plt.plot(X, y_dash, color='red')
plt.title('Linear Regression')
plt.xlabel('Engine Temperature')
plt.ylabel('Engine Pressure')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
print('x= \n', X)
print('x_poly = \n', X_poly)

lin2 = LinearRegression()
lin2.fit(X_poly, y)

plt.scatter(X, y, color='blue')
y_pred = lin2.predict(X_poly)
plt.plot(X, y_pred, color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Engine Temperature')
plt.ylabel('Engine Pressure')
plt.show()

#Predicting a new result with Linear Regression
print("LinearRegression: ", lin.predict( [[110.0]] ) )

#Predicting a new result with Polynomial Regression
print("PolynomialRegression: ", lin2.predict( poly.fit_transform([[110.0]]) ) )