import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification


# Create K-nearest-neaighbors model
def build_model(ds, n):
    X = ds[0]
    y = ds[1]
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X, y)
    return model


# Return the percent correct of the given model on the given dataset
def percent_accuracy(ds, model):
    X = ds[0]
    y = ds[1]
    h = model.predict(X)
    percent = 100*(sum(y==h)/len(y))
    return percent
        

# Return vector of correlations between each feature and the output
def correlation(ds):
    X = ds[0]
    y = ds[1]
    df1 = pd.DataFrame(X)
    df2 = pd.DataFrame(np.array([y for i in range(len(X[0]))]).T)
    corr = df1.corrwith(df2)
    return corr


# Return the indicees of the x best predictors
def best_predictors(ds, x):
    corr = pd.Series(correlation(ds))
    corr = corr.abs()
    best = corr.nlargest(x, 'all')
    best = best.index.values
    return best


def select_features(ds, best):
    X = np.transpose(ds[0])
    X_new = []
    for i in best:
        X_new.append(X[i])
    X = np.transpose(X_new)
    ds=[X, ds[1]]
    return ds

# Return a list of the ranges of each of k data subset
def k_split_ranges(ds, k):
    N = len(ds[0])
    dim = divmod(N,k)
    ranges = []
    start = 0
    for i in range(k):
        end = start + dim[0] + (dim[1]>i)
        ranges.append([start, end])
        start = end
    return ranges


# Split data into separate training and testing datasets based on a given range r
def split_data(ds, r):
    # Designate training data, outside range r (ds1)
    X1 = np.concatenate((ds[0][:r[0]], ds[0][r[1]:]))
    y1 = np.concatenate((ds[1][:r[0]], ds[1][r[1]:]))
    ds1 = [X1, y1]
    
    # Designate testing data, inside range r (ds2)
    X2 = ds[0][r[0]:r[1]]
    y2 = ds[1][r[0]:r[1]]
    ds2 = [X2, y2]
    
    return [ds1, ds2]


# Average k-fold accuracy of knn model with optional fold-wise feature filtering
def test_model(ds, k, n, x=0):
    accuracy = 0
    for r in k_split_ranges(ds, k):
        data = split_data(ds, r)
        if (x!=0):
            best = best_predictors(data[0], x)
            ds1 = select_features(data[0], best)
            ds2 = select_features(data[1], best)
        else:
            ds1=data[0]
            ds2=data[1]
        model = build_model(ds1, n)
        accuracy += percent_accuracy(ds2, model)
    return (accuracy/k)
      

    
"""
Choose parameters:
    k - number of folds in the cross-validation
    n - number of nearest neighbors considered by the KNN model
    x - number of 'best predictors' not removed from dataset
    ds - specifications of the randomly-generated dataset being tested
"""
k = 50
n = 1
x = 100 # Note: x < n_features
ds = make_classification(n_samples=50, n_features=5000)



"""
The Wrong Way:
    1. Feature selection using entire dataset
    2. Test KNN model accuracy using cross-validation
    
    Selecting the features on the basis of the entire dataset causes the
    testing data of each fold to influence the model and inflate its accuracy
"""
print("The wrong way:")
best = best_predictors(ds, x)
ds_alt = select_features(ds, best)
accuracy = test_model(ds_alt, k, n, 0)
print("Accuracy: "+str(accuracy)+"%")
print()



"""
The Right Way:
    1. Use cross vaidation to test KNN model accuracy
    2. Perform feature-selection for each individual training data "fold"
    
    This prrocess ensures that the testing data of any given "fold" in k-fold 
    cross-validation does not influence the model evaluating it
"""
print("The right way:")
accuracy = test_model(ds, k, n, x)
print("Accuracy: "+str(accuracy)+"%")
print()

