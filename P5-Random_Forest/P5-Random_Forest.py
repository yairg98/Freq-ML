import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


training = "https://raw.githubusercontent.com/yairg98/Freq-ML/master/P5-Random_Forest/OnlineNewsPopularity_training.csv"
validation = "https://raw.githubusercontent.com/yairg98/Freq-ML/master/P5-Random_Forest/OnlineNewsPopularity_validation.csv"
testing = "https://raw.githubusercontent.com/yairg98/Freq-ML/master/P5-Random_Forest/OnlineNewsPopularity_testing.csv"


# Download, normalize, and separate dataset into inputs (X) and outputs (y)
def get_data(url):
    df = pd.read_csv(url)
    y = df['Y']
    del df['Y']
    columns = list(df.columns.values)
    X = df.to_numpy()
    X = np.array([(i - min(i))/(max(i) - min(i)) for i in X.T]).T
    return [X, y, columns]


# Return baseline RMSE by predicting the average for every value
def baseline_rmse(ds):
    y = ds[1]
    avg = [np.mean(y)]*len(y)
    mse = np.sum(np.square(avg - y))/len(y)
    return np.sqrt(mse)


# Train random forest model on provided dataset
def build_model(ds, n, d):
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
def plot_feature_importance(model, labels):
    plt.figure(1)
    performance = model.feature_importances_
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Feature importance')
    plt.title('All Features')
    plt.show()