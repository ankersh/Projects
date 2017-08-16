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

print len(testlist)

labels = LabelEncoder().fit_transform(labels)           #Converts the labels into an integer so the CNN can use them
labels = np_utils.to_categorical(labels)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(convertedarray, labels, test_size=0.1,random_state=10)          #Splits training data randomly



######################       CNN           ###############################

model = Sequential()

model.add(BatchNormalization(mode=1,epsilon=0.002,input_shape=(Y,X,3)))                                           #Layer 1: Keeps mean activation of the CNN close to 1, keeps SD close to 0.
model.add(GaussianNoise(1.5))

model.add(Convolution2D(32, 3, 3,border_mode='same',dim_ordering='th'))                             #Layer 2: Simple Convolution2D with 32 output filters and relu activation
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3,border_mode='same', dim_ordering='th'))                            #Layer 3: Simple Convolution2D with 64 output filters and relu activation
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Convolution2D(128, 3, 3,border_mode='same', dim_ordering='th'))                            #Layer 4: Simple Convolution2D with 128 output filters and relu activation
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3,border_mode='same', dim_ordering='th'))                            #Layer 4: Simple Convolution2D with 256 output filters and relu activation
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))


model.add(Flatten())                                                                                #Flattens the output from 3d to 1d

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(8))                                                                                 #Fully connected layer, outputs an array with a dimension of 8 (one for every fish class)
model.add(Activation('softmax'))                                                                    #Uses sigmoid activation to output the final predictions
model.compile(loss='categorical_crossentropy', optimizer=SGD(nesterov=True))





model.fit(Xtrain,Ytrain,nb_epoch=10, batch_size=64)

preds = model.predict_proba(Xtest)
print("Validation Log Loss: " + format(log_loss(Ytest, preds)))

testpreds = model.predict_proba(testarray)
submission = pd.DataFrame(testpreds, columns=classes)
submission.insert(0,'image',testname)
submission.to_csv("submission.csv")
