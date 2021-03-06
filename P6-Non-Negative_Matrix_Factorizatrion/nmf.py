# -*- coding: utf-8 -*-
"""NMF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZAHgU5yh_0Gd_G7wTzvKsRR-VC1n_9kP

Yair Gross and Hadassah Yanofsky
August, 2020
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

data='https://raw.githubusercontent.com/HYanofsky/random-cvs/master/genome-scores-modified.csv'


n_samples=1127  #number of movies
n_features=25   #number of tags
users=10        #number of users


def make_model(X):
  model = NMF(n_components=users)
  return model


#Download the data
df = pd.read_csv(data)
V=np.array(df)          #V matrix uses data from to say the relevance each movie has to each of the tags and classify the movies in that way

model=make_model(V)

W = model.fit_transform(V)    #using the correlation between movies and tags we could see how the users will like the movies
H = model.components_         #how much the users like the different tags that are possible for the movies

print("Dim V: ", V.shape)
print("Dim W: ", W.shape)
print("Dim H: ", H.shape)

print("\nRecommendations:")
[x, y]=np.where(W==np.max(W))
print("Number ", y, " user should watch the number ", x, " movie." )