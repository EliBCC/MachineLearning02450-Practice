# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:17:12 2019

@author: Eli
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import tree
import random
import graphviz

# Percent of data used for training
# K must be <= number of rows in X
K=10

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

# Drop Vandenberg Height
X=np.delete(X,1,1)
attributeLabels = np.delete(attributeLabels,1)


# Create training and test sets
X_train=X
y_train=y
row_index=random.randint(0,np.size(X_train,0))
X_test=X_train[row_index]
y_test=y_train[row_index]
X_train=np.delete(X_train,row_index,0)
y_train=np.delete(y_train,row_index,0)
for i in range(int((np.size(X,0))*K/100)-1):
    row_index=random.randint(0,np.size(X_train,0)-1)
    X_test=np.vstack([X_test,X_train[row_index]])
    X_train=np.delete(X_train,row_index,0)
    y_test=np.vstack([y_test,y_train[row_index]])
    y_train=np.delete(y_train,row_index,0)

# Recompute size
N, M=X_train.shape

# Fit regression tree classifier, Gini split criterion, pruning enabled
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=10)
dtc = dtc.fit(X_train,y_train)

# Compute error
correct=0
for i in range(np.size(X_test,0)):
    x_class = int(dtc.predict(X_test[i].reshape(1,-1))[0])
    correct=correct + (x_class is int(y_test[i]))

print("Accuracy is: "+str(correct/np.size(X_test,0)))

out = tree.export_graphviz(dtc, out_file='LAozoneData.gvz', feature_names=attributeLabels)
src=graphviz.Source.from_file('LAozoneData.gvz')
src.render('../LAozoneData', view=True) 