# Freq ML Project 4 - Gradient Boosted Trees (with XGBoost)
This is the fourth project for Cooper Union's Frequentist Machine Learning course.
It used the XGBoost Regressor to predict house sale prices using data from the Kaggle competition linked below.
The program usesa grid search to optimize each of several model parameters, including L1 regularization.

#### Results Overview:
The mean-based baseline regressor had an RMSE of 78,837 or 43.58%.

The initial best-guess model had an RMSE of 5,601 or 3.10%.

The grid-search optimized model had an RMSE of 2,559 or 1.43%, a significant imporvement with relatively minimal tuning.

Dataset used: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/

## Assignment Description
#### Assignment 4: xTreme Gradient Boosted Trees
Read sections: 9.2, all of chapter 10 (sorry) and https://arxiv.org/pdf/1603.02754v3.pdf

Select a dataset for either classification or regression.
It can be the same dataset as the one you used in the previous assignments, or a new one.
Use an out of the box package for xTreme gradient boosting trees such as https://xgboost.readthedocs.io/en/latest/.
Don’t use sci-kit’s gradient boosting method because there is no regularization built in (this is a key difference between gradient boosting and xTreme).

Use the same 80-10-10 split to tune your classifier/regression method and report your performance and output the feature importance.
Do the features reported make sense?
If you are using the same dataset from assignment 1 or 2, do they agree with what you discovered using the Lasso penalty?
