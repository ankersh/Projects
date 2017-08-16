import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten




traind = "sampleTrain"
testd = "test"
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


imagelist = []                  #Contains all the raw images
for c in classes:
    fish = [traind+'/'+c+'/'+item for item in os.listdir(traind+'/'+c)]
    imagelist.extend(fish)

labels = []                     #Contains the label for imagelist[i] at labels[i]
for c in classes:
    l = [c for i in os.listdir(traind+'/'+c)]
    labels.extend(l)


X = 240             #Image width
Y = 120              #Image height

convertedarray = np.ndarray((len(imagelist), Y, X, 3))          #The numpy array I use to pass data to my CNN


for i in range(0,len(imagelist)):                       #Uses OpenCV to convert the image into a 3d array
    rawimage = cv2.imread(imagelist[i])
    rawimage = cv2.resize(rawimage,(X,Y))
    convertedarray[i] = rawimage


labels = LabelEncoder().fit_transform(labels)           #Converts the labels into an integer so the CNN can use them
labels = np_utils.to_categorical(labels)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(convertedarray, labels, test_size=0.3,random_state=10)          #Splits training data randomly



######################       CNN           ###############################

model = Sequential()

model.add(Convolution2D(32, 3, 3,border_mode='same',dim_ordering='th',input_shape=(Y,X,3)))         #Layer 1: Simple Convolution2D with 32 output filters and relu activation
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3,border_mode='same', dim_ordering='th'))                            #Layer 2: Simple Convolution2D with 64 output filters and relu activation
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())                                                                                #Flattens the output from 3d to 1d

model.add(Dense(8))                                                                                 #Fully connected layer, outputs an array with a dimension of 8 (one for every fish class)
model.add(Activation('sigmoid'))                                                                    #Uses sigmoid activation to output the final predictions

model.compile(loss='categorical_crossentropy', optimizer='sgd')





model.fit(Xtrain,Ytrain,nb_epoch=4, batch_size=64)

preds = model.predict(Xtest)
print("Log Loss: " + str(log_loss(Ytest, prediction)))
