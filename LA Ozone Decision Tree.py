# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:17:12 2019

@author: Eli
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import tree

#%% Import data
filename = 'LAozone.data.csv'
df = pd.read_csv(filename)

#%% Construct data matrices
raw_data = df.get_values()
y_data = raw_data[:, -1]
x_data = raw_data[:, :-2]
cols = range(0, x_data[0].size) 
attributeLabels = np.asarray(df.columns[cols])

# Define classifiers
classNames = set(y_data)
classDict = dict(zip(classNames, range(len(classNames))))
# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in y_data])

# X as continous data
X = x_data.astype(np.float64)
#Code for removing ibh
#X = np.delete(X, 5, axis=1) # Remove attribute from datamatrix
#attributeLabels = np.delete(attributeLabels, 5) # Remove label

# Compute values of N, M and C.
N = len(y)
M = len(attributeLabels)
C = len(classNames)

# Remove outliers
outlier_mask = (X[:,2]>15)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]

# Drop Vandenberg Height and doy
np.delete(X,1,1)
np.delete(X,-1,1)
attributeLabels = attributeLabels[0:11]

N, M=X.shape

# Fit regression tree classifier, Gini split criterion, pruning enabled
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=10)
dtc = dtc.fit(X,y)

out = tree.export_graphviz(dtc, out_file='LAozoneData.gvz', feature_names=attributeLabels)
src=graphviz.Source.from_file('LAozoneData.gvz')
src.render('../LAozoneData', view=True) 