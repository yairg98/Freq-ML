'''
Implement the XGBoost regression model for the Kaggle House Prices competition.
The below code uses a grid search to optimize hyperparameters.

The current method could be made more efficient by removing the less important
parameters. Or it could likely be made more accurate by expanding the parameter
search grid or trying different hyperparameter tuning methods.
However, the current process takes only a few minutes to run and achieves an
average cross-validated RMSE of ~2,600, or 1.4%. This compares very favorably 
with the baseline RMSE of over 79,000, or approximately 44%.

The model is tested on the training data vie cross validation. There is an 
acompanying test file, but is is unlabeled, as the responses are designed to
be submitted to Kaggle competition.

More info on the competition and the dataset:
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques/
'''


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.dummy import DummyRegressor


train = 'https://raw.githubusercontent.com/yairg98/Frequentist-Machine-Learning/master/P4-Gradient_Boosted_Trees/house-prices-data/train.csv'

    
# Download, clean, and format data
def get_data(url):
    # Load data
    df = pd.read_csv(url)
    
    # Separate the output column
    if 'SalePrice' in df.columns.values:
        y = df['SalePrice']
        df.drop('SalePrice', axis=1)
    else:
        y = []
    
    # One-hot-encode categorical features
    categorical = df.select_dtypes(include='object').columns.values
    # Convert numerical features to categorical
    categorical = np.append(categorical, ['MSSubClass']) # Special case
    X = pd.get_dummies(df, columns=categorical, )

    # Save ordered list of column names, just in case
    columns = list(X.columns.values)
    
    # Normalize numerical features
    X = X.to_numpy()
    X = np.array([(i - min(i))/(max(i) - min(i)) for i in X.T]).T
    
    return [X, y, columns]


# Use grid search to construct an optimal XGBoost regression model
def build_model(train_data):
    X = train_data[0]
    y = train_data[1]
    
    # Create XGBoost regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=123)
    
    # Define model parameters and test-values
    # Model gets better as n_estimators is increased, but shows dimimnishing returns as training time increases linearly
    params = {'colsample_bytree': [.5, .7, .9],
              'learning_rate': [.01, .1, .25],
              'max_depth': range(3, 12, 2),
              'reg_alpha':[0.1, 1, 100]}
    
    # Tune and train XGBoost model using SciKit grid search
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=params, 
        n_jobs = -1, 
        cv = 10, 
        verbose=True
    )
    grid_search.fit(X, y) 
    model = grid_search.best_estimator_
    return model


# Calculate a baseline RMSE by predicting the mean every time
def baseline(data):
    X = data[0]
    y = data[1]
    baseline = DummyRegressor(strategy='mean')
    score = cross_val_score(baseline, X, y, scoring='neg_root_mean_squared_error', cv=15)
    return np.average(score)


# Average RMSE of cross-validation performance
def cross_validate(model, data):
    X = data[0]
    y = data[1]
    score = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=15)
    return np.average(score)
    

#%%
'''
Build and Test the Model:
    
'''
# Load and prepare data
data = get_data(train)

# Construct, train, and optimize model
model = build_model(data)

# Establish a baseline prediction accuracy
result = round(-baseline(data), 2)
print("CV RMSE: "+str(result))
percent = round(100*result/np.average(data[1]), 2)
print("RMSE as percentage of average house sale value: "+str(percent)+'%')
print()

# Cross validate model and print final RMSE
result = round(-cross_validate(model, data), 2)
print("CV RMSE: "+str(result))
percent = round(100*result/np.average(data[1]), 2)
print("RMSE as percentage of average house sale value: "+str(percent)+'%')
