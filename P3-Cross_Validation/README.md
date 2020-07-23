# Freq ML Project 3 - Model Assessment
This is the third project for Cooper Union's Frequentist Machine Learning course. The script demonstates correct and incorrect use of k-fold cross-validation.
The first run uses the whole dataset to perform feature selection, then uses cross-validation to test the KNN model. Because all of the data was used for feature selection,
the test data is able to influence the model, and the test returns an exagerated percent accuracy.
The second run also uses cross-validation, but selects the best features based on the unique training data for each fold in the cross-validation.
This allows for feature selection without compromising the test.

The data used for this demonstration is a dummy set, generated using sklearn.datasets.make_classification(). This can be trivially replaced with real data, but results will vary.

## Assignment Description
#### Assignment 3: Model Assessment and Selection
Read sections: 7.1, 7.2, 7.3, 7.10
Re-implement the example in section 7.10.2 using any simple, out of the box classifier (like K nearest neighbors from sci-kit).
Reproduce the results for the incorrect and correct way of doing cross-validation.
