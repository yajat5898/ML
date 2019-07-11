# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Sat Jul  7 00:19:42 2018)---
runfile('C:/Users/Yajat/.spyder-py3/temp.py', wdir='C:/Users/Yajat/.spyder-py3')
runfile('C:/Users/Yajat/.spyder-py3/temp.py', wdir='C:/Users/Yajat/.spyder-py3')
runfile('C:/Users/Yajat/.spyder-py3/untitled0.py', wdir='C:/Users/Yajat/.spyder-py3')
runfile('C:/Users/Yajat/Desktop/resize.py', wdir='C:/Users/Yajat/Desktop')

## ---(Fri Jul 13 09:51:33 2018)---
runfile('C:/Users/Yajat/Desktop/ball_trackin.py', wdir='C:/Users/Yajat/Desktop')
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import serial
from collections import deque
import argparse
import cv2
import imutils
import argparse
import cv2
import time
import serial
runfile('C:/Users/Yajat/Desktop/ball_trackin.py', wdir='C:/Users/Yajat/Desktop')
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import serial
import numpy as np
import argparse
import cv2
import imutils
import time
import serial
import time
import serial

## ---(Mon Jun  3 15:09:18 2019)---
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
%clear
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
%clear
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
x
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
x
labelEncoder_x
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
x
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 3].values
# taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0 )
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
oneHotEncoder_x = OneHotEncoder(categorical_features = [0])
x= oneHotEncoder_x.fit_transform(x).toarray()
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
oneHotEncoder_x = OneHotEncoder(categorical_features = [0])
x= oneHotEncoder_x.fit_transform(x).toarray()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 3].values
# taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0 )
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x
oneHotEncoder_x = OneHotEncoder(categorical_features = [0])
x= oneHotEncoder_x.fit_transform(x).toarray()
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
runfile('C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing/data_preprocessing.py', wdir='C:/Users/Yajat/Desktop/ML/Part 1 - Data Preprocessing')
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
%varexp --imshow x
y = dataset.iloc[:,:3].values
x
y
y = dataset.iloc[:,3].values
y

## ---(Mon Jul  8 12:38:38 2019)---
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('input.csv')
x = dataset.iloc[:,:].values
dataset1 = pd.read_csv('output.csv')
y = dataset1.iloc[:,:].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
from sklearn.cross_validation import train_test_split
x_train, x_test, x_validate = np.split(x,[int(.6*len(x)),int(.8*len(x))])
y_train, y_test, y_validate = np.split(y,[int(.6*len(y)),int(.8*len(y))])
from matplotlib.colors import ListedColormap
x_set,y_set = x,y
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min -1),stop = x_set[:,0].max +1,step =0.01 ),
                                np.arange(start = x_set[:,0].min -1),stop = x_set[:,0].max +1,step =0.01 ))
plot.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label = j)


plt.show()
from matplotlib.colors import ListedColormap
x_set,y_set = x,y
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min -1),stop = x_set[:,0].max +1,step =0.01 ), np.arange(start = x_set[:,0].min -1),stop = x_set[:,0].max +1,step =0.01 ))
plot.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label = j)


plt.show()
import scipy.misc
image_data = pd.read_csv('input.csv')
print('image data read successfully')
print(image_data.loc[0:0])
print('-------------------------------------------------------------------')
row = image_data.loc[4700:4700]
a = np.array(row)
new = np.reshape(a,(20,20))
print(new)
scipy.misc.imsave('outfilenew.jpg',new)
from sklearn.model_selection import train_test_split
x_train, x_test, x_validate = np.split(x,[int(.6*len(x)),int(.8*len(x))])
y_train, y_test, y_validate = np.split(y,[int(.6*len(y)),int(.8*len(y))])
from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
import keras
import keras.models import Sequential
import keras.models import Dense
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier = Sequential(output_dim=201,init='uniform',activation='relu')
classifier.add(Dense(output_dim=201,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_dim=400))
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',))
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
classifier.fit(x_train,y_train,batch_size=10,nb_epochs=100)
classifier.fit(x_train,y_train, batch_size=10, nb_epochs=100)
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier.fit(x_train,y_train, batch_size=100, epochs=100)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,:]=labelencoder_x.fit_transform(x[:,:])
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
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_dim=400))

#Adding the second hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',))

#Adding the output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier.fit(x_train,y_train, batch_size=1, epochs=100)
image_data = pd.read_csv('input.csv')
print('image data read successfully')
print(image_data.loc[0:0])
print('-------------------------------------------------------------------')
row = image_data.loc[1000:1000]
a = np.array(row)
new = np.reshape(a,(20,20))
print(new)
scipy.misc.imsave('outfilenew.jpg',new)
classifier.fit(x_train,y_train, batch_size=1, epochs=100)
classifier.add(Dense(201,init='uniform',activation='relu',input_dim=400))
classifier.add(Dense(201,init='uniform',activation='relu',))
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.add(Dense(10,init='uniform',activation='softmax'))
classifier.fit(x_train,y_train, batch_size=1, epochs=100)
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_dim=400,input_shape=(10,)))

#Adding the second hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax',input_shape=(10,)))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the second hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax',input_shape=(10,)))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

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
model = Sequential()

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
model.fit(x_train, y_train, epochs=150, batch_size=10)
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the second hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax',input_shape=(10,)))
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the second hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax',input_shape=(10,)))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier.add(Dense(output_dim=201,init='uniform',activation='relu',input_shape=(10,)))

#Adding the second hidden layer
classifier.add(Dense(output_dim=201,init='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax',))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(200,init='uniform',activation='relu',input_shape=(400,)))

#Adding the second hidden layer
classifier.add(Dense(200,init='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(10,init='uniform',activation='softmax',))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(200,init='uniform',activation='relu',input_shape=(400,)))

#Adding the second hidden layer
classifier.add(Dense(200,init='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(10,init='uniform',activation='softmax'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(200,init='uniform',activation='relu'))

#Adding the second hidden layer
classifier.add(Dense(200,init='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(10,init='uniform',activation='softmax'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
classifier.add(Dense(200,init='uniform',activation='relu'))
classifier.add(Dense(201,init='uniform',activation='relu',input_shape=(400,1)))
classifier.add(Dense(201,kernel_initializer='uniform',activation='relu',input_shape=(400,1)))
classifier.add(Dense(201,kernel_initializer='uniform',activation='relu',input_shape=(400,1)))

#Adding the second hidden layer
classifier.add(Dense(200,kernel_initializer='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(10,kernel_initializer='uniform',activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train,y_train, batch_size=10, epochs=100)