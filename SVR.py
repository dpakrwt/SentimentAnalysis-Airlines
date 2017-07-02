#SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
                
#fitting the dataset to the SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #gaussian kernel
regressor.fit(x,y)
               
#visualizing SVR
pt.scatter(x,y,color='red')
pt.plot(x,regressor.predict(x),color='green')
pt.title("Truth or Bluff(SVR)")
pt.xlabel("Position")
pt.ylabel("Salary")
pt.show()

#predicting a new result using SVR model
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))