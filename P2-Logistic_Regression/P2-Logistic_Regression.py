import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


training = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P2-Logistic_Regression/banknote_authentication_training.csv'
validation = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P2-Logistic_Regression/banknote_authentication_validation.csv'
testing = 'https://raw.githubusercontent.com/yairg98/Freq-ML/master/P2-Logistic_Regression/banknote_authentication_testing.csv'


# Normalize data by scaling to range (0,1)
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
  X = normalize(X)
  # Add leading column of 1s:
  new_col = [1]*len(X)
  X = np.insert(X, 0, new_col, axis=1)

  return [X, y, columns]


# Predict outputs for data (X) based on the provided theta
def predict(X, theta):
    # y = 1/(1 + e^(-ϴX))
    y = 1/(1 + np.exp(-np.matmul(theta,X.T)))
    return y


# Compare the predicted and actual values and return the percent correct
def accuracy(h, y):
    correct = 0
    for i in range(len(y)):
        if y[i] == round(h[i]):
            correct += 1
        else:
            #print("Wrong: "+str(round(h[i],2))+", Loss: "+str(loss(h[i],y[i])))
            pass
    accuracy = correct/len(y)
    return accuracy*100


# Returns the log loss for a given predicted/actual value pair
def loss(h, y):
    if y==1:
        return -math.log(h, 10)
    else:
        return -math.log(1-h, 10)


# Returns the net log cost of a set of predictions
def cost(h, y):
    C = 0
    for i in range(len(y)):
        C += loss(h[i], y[i])
    C = C/len(y)
    return C


# Returns the log cost given a given lambda and theta, on the given data
def getCost(X, y, theta, lamda=0):
    h=predict(X, theta)
    m = y.shape[0]
    C = cost(h, y)
    penalty = np.sum(theta**2)*(lamda/(2*m)) # add L2 penalty (ridge)
    C += penalty
    return C


# Batch gradient descent, returns theta vector
def BGD(ds, alpha, max_iter):
    X = ds[0]
    y = ds[1]
    theta = np.zeros(len(X[0]))
    for n in range(max_iter):
        h = predict(X, theta)
        delta = (1/len(X))*alpha*np.matmul(np.subtract(y, h), X)
        theta = np.add(theta, delta)
    return theta


# Stochastic gradient descent, returns theta vector
def SGD(ds, alpha, max_iter, lamda=0):
    X = ds[0]
    y = ds[1]
    indices = [*range(len(X))]
    theta = np.zeros(len(X[0]))
    for n in range(max_iter):
        random.shuffle(indices)
        for i in indices:
            h = predict(X[i], theta)
            delta = np.subtract(alpha*(y[i]-h)*X[i], theta*lamda)
            theta = np.add(theta, delta)
    return theta


# Mini-Batch gradient descent, returns theta vector
def MBGD(ds, alpha, max_iter, batch_size, lamda=0):
    X = ds[0]
    y = ds[1]
    indices = [*range(len(X))]
    theta = np.zeros(len(X[0]))
    for n in range(max_iter):
        batch = random.sample(indices, batch_size)
        Xn = np.array([X[i] for i in batch])
        yn = [y[i] for i in batch]
        h = predict(Xn, theta)
        delta = (1/len(Xn))*alpha*np.matmul(np.subtract(yn, h), Xn)
        theta = np.add(theta, delta)
    return theta


#Preform SGD with different lambdas and find the percent accuracy after each iteration
def SGD_find_percent(ds, alpha, max_iter, percent, lamda=0):
    X = ds[0]
    y = ds[1]
    theta = np.zeros(len(X[0]))
    for n in range(max_iter):
        # for i in random.sample(range(len(X)), k=batch_size)
        for i in range(len(X)):
            y_hat = predict(X[i], theta)
            delta = alpha*((y[i]-y_hat)*X[i])
            temp = np.add(theta, delta)
            theta = np.subtract(temp, np.dot(lamda, theta))
        percent[n]=testModel(ds, theta)
    return theta
        

# return the percent model accuracy given theta, on the provided dataset
def testModel(ds, theta):
    X = ds[0]
    y = ds[1]
    y_hat = predict(X, theta)
    percent = accuracy(y_hat, y)
    return str(round(percent, 2))


# Find theta and return the cost for the found cost and the given lambda 
def tryModel_SGD(ds1, ds2, alpha, max_iter, lamda):
    data_1 = getData(ds1)
    data_2 = getData(ds2)
    theta = SGD(data_1, alpha, max_iter, lamda)
    cost = getCost(data_2[0], data_2[1], theta, lamda)
    return cost 


# Find the lambda that has the lowest error (log cost)
def bestLam_SGD(ds1, ds3, alpha, max_iter, S):
    lam = S[0]
    min_err = tryModel_SGD(ds1, ds3, alpha, max_iter, lam)
    for i in S:
        err = tryModel_SGD(ds1, ds3, alpha, max_iter, i)
        if err < min_err:
            min_err = err
            lam = i
    return lam



'''
Project 2 - Logistic Regression With Stochastic Gradient Descent:
    
Choose the alpha and max_iter parameters for the below models. Increasing 
max_iter will logarithmically improve the accuracy of the model while 
increasing training time. Lowering alpha will improve the model's precision, 
but increase the number of iterations until convergence.
'''
# Choose model parameters
alpha = 0.5
max_iter = 10

# Download all the data
ds1 = getData(training)
ds2 = getData(testing)
ds3 = getData(validation)

'''
Part A - Unregularized Stochastic Gradient Descent:
    The model already performs reasonably well (90+ percent) with as few as 3 
    iterations (max_iter=3). That percentage continues to improve, approaching
    100% asymptotically, as max_iter is increased. This also increases training
    time. Alpha can be incresed for faster convergence with lower accuracy.
'''
print("Part A:")
# Find percent accuracy with lambda=0
no_lam=np.zeros(max_iter)
print("Accuracy of unregularized SDG model:")
theta = SGD_find_percent(ds1, alpha, max_iter, no_lam)
print(testModel(ds1, theta), "% correct [training]")
print(testModel(ds2, theta), "% correct [testing]")
print("Cost: ", getCost(ds2[0], ds2[1], theta))
print()

'''
Part B - SGD With L2 Regularization:
    Regularization was found not to have any positive impact on this model and 
    dataset at any lambda value. This makes sense given the relatively small
    number of features. For convenient comparison, alpha and max_iter are the
    same as in part a.
'''
print("Part B:")
# Find best lambda
S = np.linspace(0, .09, 10)
lam = bestLam_SGD(training, validation, alpha, max_iter, S)
print("Best lambda is ", lam)

# Find the percent accuaracy with ideal lambda
with_lam=np.zeros(max_iter)
print("Using ideal lam:")
theta = SGD_find_percent(ds1, alpha, max_iter, with_lam, lam)
print(testModel(ds1, theta), "% correct [training]")
print(testModel(ds2, theta), "% correct [testing]")
print("Cost: ", getCost(ds2[0], ds2[1], theta, lam))

# Plot the percent accuaracy over the different iterations
# Because lamda=0, the lines on the graph are directly on top of one another.
x=range(max_iter)
plt.figure(1)
plt.plot(x, with_lam, label="λ is optimized")
plt.plot(x, no_lam, label="λ = 0")
plt.xlabel('Iteration Number')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy WRT # of Iterations')
plt.legend()
plt.show()
