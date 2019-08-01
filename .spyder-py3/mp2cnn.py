# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:30:23 2019

@author: Yajat
"""

from keras import backend as K
K.clear_session()

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialization
classifier = Sequential()

#convolution
classifier.add(Convolution2D(64,(3,3),input_shape = (28,28,1),activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,(3,3),activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Flattening
classifier.add(Flatten())

#Full Connection
#input layer
classifier.add(Dense(128,activation = 'relu'))

#output layer
classifier.add(Dense(10,activation = 'softmax'))

#Compiling the CNN
classifier.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fitting the CNN to the  image
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()



x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)
input_shape=(28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes=10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




classifier.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis = 1),y_pred.argmax(axis=1))


from sklearn.metrics import accuracy_score
a_s = accuracy_score(y_test.argmax(axis = 1),y_pred.argmax(axis=1))
print('Accuracy:',a_s)

from sklearn.metrics import classification_report
c_r = classification_report(y_test.argmax(axis = 1),y_pred.argmax(axis=1))
print('Report:')
print(c_r)




