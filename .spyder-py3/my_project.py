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
data = pd.read_csv('input.csv')
labels = pd.read_csv('output.csv')




#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, x_validate = np.split(data,[int(.6*len(data)),int(.8*len(data))])
y_train, y_test, y_validate = np.split(labels,[int(.6*len(labels)),int(.8*len(labels))])

#Building ANN model
#Importing Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense


#Encoding categorical data
num_classes = 11
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(32,kernel_initializer='uniform',activation='relu',input_shape=(400,)))

#Adding the second hidden layer
classifier.add(Dense(32,kernel_initializer='uniform',activation='relu'))
  
#Adding the output layer
classifier.add(Dense(11,kernel_initializer='uniform',activation='softmax'))

classifier.summary()

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)

#Making the predictions and evaluating the results
#Predicting the value for y
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0)
#Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis = 1),y_pred.argmax(axis=1))















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


    












