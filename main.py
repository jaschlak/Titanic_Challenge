# -*- coding: utf-8 -*-
"""
Titanic Challenge powered by Kaggle

This is a challenge put up by kaggle, the objective is to 
predict if people will survive the Titanic crash based
on known data before the crash.

Started June 6, 2021

Conspiring Authors:
Jordan Schlak
Paul Teeter
"""

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


train_data = pd.read_csv('Input/train.csv')
test_data = pd.read_csv('Input/test.csv')

# %% import data
dataset = train_data

# organize columns to produce X and y
cols = list(dataset.columns)
y_col = ['Survived']
cols.pop(cols.index('Survived'))
x_col = cols

X = dataset[x_col].values
y = dataset[y_col].values

# %% Encoding

# encode pclass
# passthrough allows other columns to be left untouched
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
