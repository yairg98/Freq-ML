# Freq ML Project 1 - Linear Regression
This is the first project for Cooper Union's Frequentist Machine Learning course, ECE-475. The official assignment description is below. 
It involved implementing and comparing three simple linear regression models: unreglarized, ridge, and lasso.

Dataset used: https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength

## Assignment Description
Linear Regression
Read sections 3.1 -3.2.1, 3.3- 3.4.3 of Elements of Satistical Learning (can be found here: https://web.stanford.edu/~hastie/ElemStatLearn/download.html)

Grab a dataset of interest from the UCI repository (or another repository), but make sure it is one that is good for regression. This means it's got numerical(not categorical) features, and the target is a continuous number.

Divide your data into roughly 80% train, 10% validation, 10% test. You must keep this split for all 3 parts of this assignment in order to compare the methods fairly.  Perform 3 flavors of linear regression:

a) Plain old linear regression, with no regularization. You must code this one by hand (i.e use equation 3.6 to find the betas).  Report the mean squared error on the test dataset.

b) Ridge regression. You must also code this one by hand(eq 3.44 to find the betas). Select the optimal value of Lambda by cross-validation using the validation dataset. Report the mean squared error on the test dataset, using the best lambda you found on the validation set. DO NOT USE THE TEST DATASET TO CHOOSE LAMBDA.

c) Lasso regression: Use one of the built in packages in sci-kit learn or MATLAB to do a Lasso regression. Select the optimal value of lambda as in part b) and also display a Lasso plot (there are built in functions for Lasso plot in sci-kit/MATLAB). Which features did the Lasso select for you to include in your model? Do these features make sense?

Compute the MSE on the training dataset and the test dataset for all methods and comment on the results.  

Feeling brave? Do Lasso and Ridge plots(like figures 3.8 and 3.10).
