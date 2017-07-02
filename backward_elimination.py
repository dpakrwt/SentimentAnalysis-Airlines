import numpy as np
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
x[:,3] = labelEncoder.fit_transform(x[:,3])
onehotEncoder = OneHotEncoder(categorical_features=[3])
x = onehotEncoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x = x[:,1:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

#using backward elimination for feature selection
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1), dtype=int), values = x, axis = 1)

x_opt1 = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt1).fit()
regressor_OLS.summary()

x_opt2 = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt2).fit()
regressor_OLS.summary()

x_opt3 = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt3).fit()
regressor_OLS.summary()

x_opt4 = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt4).fit()
regressor_OLS.summary()

x_opt5 = x[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt5).fit()
regressor_OLS.summary()