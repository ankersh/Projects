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
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.optimizers import SGD
from keras.layers.noise import GaussianNoise

trainfolder = "train"
testfolder = "test"
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


imagelist = []                  #Contains all the raw images
for c in classes:
    fish = [trainfolder+'/'+c+'/'+item for item in os.listdir(trainfolder+'/'+c)]
    imagelist.extend(fish)

labels = []                     #Contains the label for imagelist[i] at labels[i]
for c in classes:
    l = [c]*len(os.listdir(trainfolder+'/'+c))
    labels.extend(l)


testlist = []
testname = []
for item in os.listdir(testfolder+'/'):
    testlist.append(testfolder+'/'+item)
    testname.append(item)

X = 256             #Image width
Y = 144              #Image height

convertedarray = np.ndarray((len(imagelist), Y, X, 3))          #The numpy array I use to pass data to my CNN


for i in range(0,len(imagelist)):                       #Uses OpenCV to convert the image into a 3d array
    rawimage = cv2.imread(imagelist[i])
    rawimage = cv2.resize(rawimage,(X,Y))
    convertedarray[i] = rawimage

testarray = np.ndarray((len(testlist), Y, X, 3))          #The numpy array I use to pass data to my CNN
for i in range(0,len(testlist)):                       #Uses OpenCV to convert the image into a 3d array
    rawimage = cv2.imread(testlist[i])
    rawimage = cv2.resize(rawimage,(X,Y))
    testarray[i] = rawimage


labels = LabelEncoder().fit_transform(labels)           #Converts the labels into an integer so the CNN can use them
labels = np_utils.to_categorical(labels)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(convertedarray, labels, test_size=0.01,random_state=10)          #Splits training data randomly



######################       CNN           ###############################

model = Sequential()

model.add(BatchNormalization(mode=1,epsilon=0.002,input_shape=(Y,X,3)))


model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(Xtrain,Ytrain,nb_epoch=12, batch_size=64)


model.save('vgg3.h5')

#preds = model.predict_proba(Xtest)
#print("Validation Log Loss: " + format(log_loss(Ytest, preds)))

testpreds = model.predict_proba(testarray)
submission = pd.DataFrame(testpreds, columns=classes)
submission.insert(0,'image',testname)
submission.to_csv("submission3.csv")
