# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:39:03 2019

@author: Yajat
"""

#DATA PREPROCESSING
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('input.csv')
x = dataset.iloc[:,:].values
dataset1 = pd.read_csv('output.csv')
y = dataset1.iloc[:,:].values

#Taking care of Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

""" #Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])


onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y) """

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, x_validate = np.split(x,[int(.6*len(x)),int(.8*len(x))])
y_train, y_test, y_validate = np.split(y,[int(.6*len(y)),int(.8*len(y))])
