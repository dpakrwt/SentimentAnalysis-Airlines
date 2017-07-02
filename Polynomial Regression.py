#polynomial regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
                
#fitting the dataset to the linear model
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(x,y)

#fitting the dataset to the polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
x_poly = poly_regressor.fit_transform(x,y)
lin2_regressor = LinearRegression()
lin2_regressor.fit(x_poly,y)

#visualizing linear regression
pt.scatter(x,y,color='red')
pt.plot(x,lin_regressor.predict(x),color='green')
pt.title("Truth or Bluff(Linear regression)")
pt.xlabel("Position")
pt.ylabel("Salary")
pt.show()

#visualizing polynomial regression
pt.scatter(x,y,color='red')
pt.plot(x,lin2_regressor.predict(poly_regressor.fit_transform(x)),color='green')
pt.title("Truth or Bluff(Polynomial regression)")
pt.xlabel("Position")
pt.ylabel("Salary")
pt.show()

#predicting a new result using linear regression model
lin_regressor.predict(6.5)

#predicting a new result using polynomial regression model
lin2_regressor.predict(poly_regressor.fit_transform(6.5))