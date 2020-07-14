# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:43:32 2020

@author: yairg
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model


training = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P1-Linear_Regression/Construction_Training.csv'
validation = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P1-Linear_Regression/Construction_Validation.csv'
testing = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P1-Linear_Regression/Construction_Testing.csv'


# Normalize all data to the range (0,1) - (all features are always >=0)
def normalize(X):
    for j in range(len(X[0])):
        high = X[0][j]
        low = X[0][j]
        for i in X:
            high = max(high, i[j])
            low = min(low, i[j])
        for i in X:
            i[j] = (i[j]-low)/(high-low)
    return X


# Download and separate dataset into input features (X) and outputs (y)
def getData(url):
  # Retrieve data from Github and store in Pandas Dataframe
  df = pd.read_csv(url)

  # Specify the output column ('charges', in this case) 
  y = df['Y']
  del df['Y']
  
  columns = list(df.columns.values)
  
  # Define feature matrix
  X = df.to_numpy()
  # Normalize the input data:
  X = normalize(X)
  # Add leading column of 1s:
  new_col = []
  for i in range(len(X)):
    new_col.append(1)
  X = np.insert(X, 0, new_col, axis=1)

  return [X, y, columns]


# Find the betas given a dataset and, optionally, lamda (for ridge regression)
def getBeta_ridge(ds1, lamda=0):
  X = ds1[0]
  y = ds1[1]
  # Calculate betas using equation 3.44
  Xt = X.transpose()
  I = lamda*np.identity(len(X[0]))
  m = np.matmul(np.linalg.inv(np.add(np.matmul(Xt, X), I)), Xt)
  beta = np.matmul(m, y)
  return beta


# Find the betas for given dataset and lamda using sklearn's Lasso model
def getBeta_lasso(ds1, lamda):
    x = ds1[0]
    y = ds1[1]
    lasso = linear_model.Lasso(alpha=lamda) # create model
    lasso.fit(x, y) # train model
    beta = lasso.coef_
    beta[0] = lasso.intercept_
    return beta


# Calcualte RMSE of linear regression model on ds1 using provided betas
def getRMSE(ds1, beta):
  # Load validation/testing data
  X = ds1[0]
  y = ds1[1]

  # Calclate RSS and MSE using equation 3.43
  Xb = np.matmul(X, beta)
  err = np.subtract(y, Xb)
  err_t = err.transpose()
  RSS = np.matmul(err_t, err)
  MSE = RSS/len(y)
  RMSE = math.sqrt(MSE)
  return RMSE


# Return RMSE of ridge regression model with provided lamda and datasets
def tryModel_ridge(ds1, ds2, lamda=0):
    beta = getBeta_ridge(ds1, lamda) # Train model on ds1
    RMSE = getRMSE(ds2, beta) # Get RMSE of model on ds2
    return RMSE


# Return RMSE of lasso regression model with provided lamda and datasets
def tryModel_lasso(ds1, ds2, lamda):
    beta = getBeta_lasso(ds1, lamda)
    RMSE = getRMSE(ds2, beta)
    return RMSE  


# Evaluate ridge regression model on ds2 for each lamda in S
def bestLam_ridge(ds1, ds3, S):
    min_err = tryModel_ridge(ds1, ds3, S[0])
    for i in S:
        err = tryModel_ridge(ds1, ds3, i)
        if err < min_err:
            min_err = err
            lam = i
    return lam


# Evaluate ridge regression model on ds2 for each lamda in S
def bestLam_lasso(ds1, ds3, S):
    lam = S[0]
    min_err = tryModel_lasso(ds1, ds3, lam)
    for i in S:
        err = tryModel_lasso(ds1, ds3, i)
        if err < min_err:
            min_err = err
            lam = i
    return lam        



'''
Part A - Unregulated linear regression:
    Using the same function as for Ridge regression, but allowing lamda to
    default to zero, thereby functioning as an unregulated regression
'''
print("Part A - Unregularized Linear Regression:")
ds1 = getData(training) # load training set
ds2 = getData(testing) # load testing set
print("RMSE of the unregularized linear regression model on the testing set: " 
      + str(tryModel_ridge(ds1, ds2)))
print("RMSE of the unregularized linear regression model on the training set: " 
      + str(tryModel_ridge(ds1, ds1)))
print()



'''
Part B - Ridge regression:
    Testing for lamda between 0 and 20 (increments of 1).
    Surprisingly, the performance of te model on the testing set seems be worse
    after regularization.
'''
print("Part B - Ridge Regression:")
ds3 = getData(validation) # load validation set
S = range(20)
lam = bestLam_ridge(ds1, ds3, S)
print("Best lamda as evaluated on validation set: " + str(lam))
print("RMSE on testing set, using lamda=" + str(lam) + ": "
      + str(tryModel_ridge(ds1, ds2, lam)))
print("RMSE on training set, using lamda=" + str(lam) + ": "
      + str(tryModel_ridge(ds1, ds1, lam)))
print()

# Creating Ridge plot
beta_mat = [] # Matrix of all beta coefficients for all tested lamdas
for lam in S:
    beta_mat.append(getBeta_ridge(ds1, lam))
beta_mat = np.array(beta_mat).T.tolist() # Transpose the beta matrix
plt.figure(1)
for x in beta_mat:
    plt.plot(S, x)
plt.xlabel('λ')
plt.ylabel('Coefficients')
plt.title('Part B: Ridge Plot')
columns = ds1[2]
plt.legend(labels=columns)



'''
Part C - Lasso regression:
    Testing for log(lamda) at 100 values between -5 and 5.
    Here too, the performance of the model seems to be worse after performing
    lasso regularization.
'''
print("Part C - Lasso regression:")
S = np.linspace(-5, 5, num=100) # range and number of tested lamda values
bestLam = bestLam_lasso(ds1, ds3, 10**S)
print("Best lamda as evaluated on validation set: " + str(bestLam))
print("RMSE on testing set, using lamda=" + str(bestLam) + ": "
      + str(tryModel_lasso(ds1, ds2, bestLam)))
print("RMSE on training set, using lamda=" + str(bestLam) + ": "
      + str(tryModel_lasso(ds1, ds1, bestLam)))
print()

# Creating Lasso plot
beta_mat = [] # Matrix of all beta coefficients for all tested lamdas
for lam in 10**S:
    beta_mat.append(getBeta_lasso(ds1, lam))
beta_mat = np.array(beta_mat).T.tolist() # Transpose the beta matrix
plt.figure(2)
for x in beta_mat:
    plt.plot(S, x)
plt.xlabel('log(λ)')
plt.ylabel('Coefficients')
plt.title('Part B: Lasso Plot')
plt.legend(labels=columns)


# Creating Predicted vs Actual y_out plot
plt.figure(3)
y_real = ds2[1]
y_out = np.matmul(ds2[0], getBeta_lasso(ds1, bestLam))
plt.plot(y_real, y_out, 'o')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs Predicted Y-Values')
m, b = np.polyfit(y_real, y_out, 1)
plt.plot(y_real, m*y_real + b)
caption = "Line of best fit: y = "+str(round(m,2))+"x + "+str(round(b,2))
plt.text(8,65,caption)


'''
Comments:
    
    Neither Ridge nor Lasso regularization improved upon the initial model
    (unregularized). An explanation for this may be suggested by another
    peculiar aspect of the data, that each model performed only slightly better
    on the testing data than on the training data. Whereas regularization was
    used to avoid overfitting, poor performance on the training data would 
    suggest that our linear model was actually underfitting the data. This 
    could be due to (1) an incomplete/insufficient feature set, (2) the data
    being partially random/non-deterministic, or (3) the data being inherently
    non-linear.
    
    That being said, all three models did show a clear correlation between the
    actual and the predicted outputs, even if there was a lot of noise.    
'''