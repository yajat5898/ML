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
import scipy.misc

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
from sklearn.model_selection import train_test_split
x_train, x_test, x_validate = np.split(x,[int(.6*len(x)),int(.8*len(x))])
y_train, y_test, y_validate = np.split(y,[int(.6*len(y)),int(.8*len(y))])

#Feature Scaling
from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Building ANN model
#Importing Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(201,kernel_initializer='uniform',activation='relu',input_shape=(400,)))

#Adding the second hidden layer
classifier.add(Dense(201,kernel_initializer='uniform',activation='relu'))
  
#Adding the output layer
classifier.add(Dense(10,kernel_initializer='uniform',activation='softmax'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)





#Copied from onlinemodel = Sequential()

# Add the first layer
# input_dim=   has to be the number of input variables. 
# It represent the number of inputs in the first layer,one per column 
model.add(Dense(12, input_dim=5, activation='relu'))

# Add the second layer
model.add(Dense(8, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)














#Visualizing the data
image_data = pd.read_csv('input.csv')
print('image data read successfully')
print(image_data.loc[0:0])
print('-------------------------------------------------------------------')
row = image_data.loc[1000:1000]
a = np.array(row)
new = np.reshape(a,(20,20))
print(new)
scipy.misc.imsave('outfilenew.jpg',new)


    












