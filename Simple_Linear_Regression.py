#Simple linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('G:\\Data Science\\working_dir\\Simple_Linear_Regression\\Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
                
#splitting into test and training set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#fitting the model to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)

#visualization of the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience (yrs)')
plt.ylabel('Salary ($)')
plt.show()


#visualization of the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'green')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience (yrs)')
plt.ylabel('Salary ($)')
plt.show()