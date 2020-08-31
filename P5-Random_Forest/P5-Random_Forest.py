import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt


# Download, normalize, and separate dataset into inputs (X) and outputs (y)
def get_data(url):
    df = pd.read_csv(url)
    y = df['Y']
    del df['Y']
    columns = list(df.columns.values)
    X = df.to_numpy()
    X = np.array([(i - min(i))/(max(i) - min(i)) for i in X.T]).T
    return [X, y, columns]


# # Return baseline RMSE by predicting the average for every value
# def baseline_rmse(ds):
#     y = ds[1]
#     avg = [np.mean(y)]*len(y)
#     mse = np.sum(np.square(avg - y))/len(y)
#     return np.sqrt(mse)


# Return baseline RMSE by predicting the average for every value
def baseline_rmse(ds1, ds2):
    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr.fit(ds1[0], ds1[1])
    h = dummy_regr.predict(ds2[0])
    return RMSE(h, ds2[1])


# Train random forest model on provided dataset
def build_model(ds, n=100, d=None):
    model = RandomForestRegressor(n_estimators=n, max_depth=d) # Optimize parameter
    model.fit(ds[0], ds[1])
    return model


# Return predictions of the given model for for the given samples X
def predict(model, X):
    y_hat = model.predict(X)
    return y_hat


# Return the RMSE given the predicted and actual outputs, y_hat and y
def RMSE(y_hat, y):
    mse = np.sum(np.square(y_hat - y))/len(y)
    rmse = np.sqrt(mse)
    return rmse


# Plot the feature importances of the model (labels = ds[2])
def plot_feature_importance(model, labels=None):
    plt.figure()
    performance = model.feature_importances_
    y_pos = np.arange(len(performance))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')
    plt.show()


# Plot data distribution with optional logarithmic scale on the y_axis
def plot_data_distribution(y, y_log=False):
    plt.figure()
    plt.hist(y, color = 'blue', edgecolor = 'black', bins=15, log=y_log)
    plt.title('Data Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.show()





"""
The sklearn random forest regressor was fairly simple to use and to tune.
However, the results were consistently bad, and only continued to deteriorate
as the number of trees and max depth increased. The issue was determined to be 
an extremely imbalanced underlying data distribution. In lieu of discussing the
model itself, this section will demonstrate the difficulties that the dataset 
posed, how they were analyzed, and what steps could be taken to handle them.
"""

# Import the data
training = "https://raw.githubusercontent.com/yairg98/Freq-ML/master/P5-Random_Forest/OnlineNewsPopularity_training.csv"
validation = "https://raw.githubusercontent.com/yairg98/Freq-ML/master/P5-Random_Forest/OnlineNewsPopularity_validation.csv"
testing = "https://raw.githubusercontent.com/yairg98/Freq-ML/master/P5-Random_Forest/OnlineNewsPopularity_testing.csv"
ds1 = get_data(training)
ds2 = get_data(testing)


# # Establish a baseline acccuracy by predicting the average value for all inputs
# baseline = baseline_rmse(ds1,ds2)
# print("1. Baseline RMSE on testing set: "+str(round(baseline,2)))
# print()


# # Train and test the random forest model on the chosen dataset
# #   Note: parameter tuning is not shown here because it takes a while and was
# #   ultimately ineffective, failing to find and reasonably effective RF model
# model = build_model(ds1, 100, 20)
# h = predict(model, ds2[0])
# rf_rmse = RMSE(h, ds2[1])
# print("2. Random forest RMSE on testing set: "+str(round(rf_rmse,2)))
# print()


# Test the same random forest and baseline models on a comparable dummy dataset
dd1 = make_regression(n_samples=32000, n_features=60, random_state=26)
dd2 = make_regression(n_samples=3500, n_features=60, random_state=26)

baseline = baseline_rmse(dd1,dd2)
print("3. Baseline RMSE on dummy data: "+str(round(baseline,2)))
print()

model = build_model(dd1, 10000, None)
h = predict(model, dd2[0])
rf_rmse = RMSE(h, dd2[1])
print("4. Random forest RMSE on dummy data: "+str(round(rf_rmse,2)))
print()




