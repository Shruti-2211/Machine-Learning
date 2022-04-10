# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:49:33 2021

"""

#import package
import numpy as np
from sklearn.linear_model import LinearRegression

#Generate Data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

#print data
print(x)
print(y)

print(x.shape)
print(y.shape)

#Train the model

#Method 1:
model = LinearRegression()
model.fit(x, y)

#Method 2:
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

#predict
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')

#generate new data
x_new = np.arange(5).reshape((-1, 1))
print(x_new)

#predict response for new data
y_new = model.predict(x_new)
print(y_new)

#plot the data set
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, model.predict(x), color = "green")
plt.show()

plt.scatter(x_new, y_new)
plt.plot(x_new, model.predict(x_new), color = "green")
plt.show()



#Same implementation with diabetes data set
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
