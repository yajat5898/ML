# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:46:17 2019

@author: Yajat
"""

import numpy as np

import pandas as pd
import scipy.misc
import keras
from keras.models import Sequential
from keras.layers import Dense

#Importing the dataset
from keras.datasets import mnist




#Splitting the dataset into the Training set and Test set
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# Flatten the images
ivs = 28*28
x_train = x_train.reshape(x_train.shape[0], ivs)
x_test = x_test.reshape(x_test.shape[0], ivs)

#Building ANN model
#Importing Keras Libraries and Packages

import keras
from keras.models import Sequential
from keras.layers import Dense



#Encoding categorical data
num_classes = 10
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(32,kernel_initializer='uniform',activation='relu',input_shape=(ivs,)))

#Adding the second hidden layer
classifier.add(Dense(32,kernel_initializer='uniform',activation='relu'))
  
#Adding the output layer
classifier.add(Dense(10,kernel_initializer='uniform',activation='softmax'))

classifier.summary()

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=10)

#Making the predictions and evaluating the results
#Predicting the value for y
y_pred = classifier.predict(x_test)
#Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis = 1),y_pred.argmax(axis=1))


from sklearn.metrics import accuracy_score
a_s = accuracy_score(y_test.argmax(axis = 1),y_pred.argmax(axis=1))
print('Accuracy:',a_s)

from sklearn.metrics import classification_report
c_r = classification_report(y_test.argmax(axis = 1),y_pred.argmax(axis=1))
print('Report:')
print(c_r)