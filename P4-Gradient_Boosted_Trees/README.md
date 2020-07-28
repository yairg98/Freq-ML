# Freq ML Project 4 - Gradient Boosted Trees (with XGBoost)
This is the fourth project for Cooper Union's Frequentist Machine Learning course.
It used the XGBoost classifier, optimizing alpha for L1 (Lasso) regularization.
The simple demo can be trivially extended to L2 reguarization.

Planning to add a grid search to optimize other parameters of XGBClassifier, replace the dataset with something larger/more complex, 
and add plots of feature importance and/or accuracy with respect to alpha or other parameters

Dataset used:

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
