"""
Notes: try grid search to tune other xgb parameters, and swap out dataset for
something more interesting/complicated (more features) to plot feature
importance with respect to alpha and lambda
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# Download, normalize, and separate dataset into inputs (X) and outputs (y)
def get_data(url):
  df = pd.read_csv(url)
  y = df['Y']
  del df['Y']      
  columns = list(df.columns.values)
  X = df.to_numpy()
  X = np.array([(i - min(i))/(max(i) - min(i)) for i in X.T]).T
  return [X, y, columns]


# Train XGB classifier with optional L1 (alpha) or L2 (lambda) regularization
def build_model(ds, alpha=0, lam=1):
    X = ds[0]
    y = ds[1]
    model = XGBClassifier(reg_alpha=alpha, reg_lambda=lam)
    model.fit(X, y)
    return model


# Return the percent accuracy of the model on the given dataset
def percent_accuracy(ds, model):
    X = ds[0]
    y = ds[1]
    h = model.predict(X)
    correct = sum(y==h)
    percent = 100*correct/len(y)
    return percent


def test_model(ds1, ds2, alpha=0, lam=1):
    model = build_model(ds1, alpha, lam)
    return percent_accuracy(ds2, model)


# Return accuracies of XGB classifier with given L1 alpha values on ds2
def test_alphas(ds1, ds2, S):
    results = []
    for a in S:
        results.append(test_model(ds1, ds2, alpha=a))
    return results



"""
The below section optimizes alpha, the L1 regualrization parameter, for an 
XGBoost classifier. In the below code, 'alpha' can be trivially replaced by 
'lam' to use L2 regularization instead of L1.
"""


# Designating data for training, validation, and testing
training = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P2-Logistic_Regression/banknote_authentication_training.csv'
validation = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P2-Logistic_Regression/banknote_authentication_validation.csv'
testing = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P2-Logistic_Regression/banknote_authentication_testing.csv'

# Selecting the range of alpha values to test
S = np.linspace(0,10,101)

# Loading all datasets
ds1 = get_data(training)
ds2 = get_data(validation)
ds3 = get_data(testing)

# Optimizing alpha
results = test_alphas(ds1, ds2, S)
best_alpha = S[np.argmax(results)]
best_result = test_model(ds1, ds3, alpha=best_alpha)

print("Best alpha: "+str(round(best_alpha, 2)))
print("Best accuracy: "+str(round(best_result, 2))+"%")