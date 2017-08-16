import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping




import theano
theano.config.openmp = True


train_dir = "train"
test_dir = "test"
classes = sorted(os.listdir(train_dir))[0:]
print classes

imagelist = []                  #Contains all the images
for c in classes:
    fish = [train_dir+'/'+c+'/'+item for item in os.listdir(train_dir+'/'+c)]
    imagelist.extend(fish)

labels = []                     #Contains the label for imagelist[i] at labels[i]
for c in classes:
    l = [c]*len(os.listdir(train_dir+'/'+c))
    labels.extend(l)



X = 240
Y = 120


convertedarray = np.ndarray((len(imagelist), Y, X, 3))


for i in range(0,len(imagelist)):
    rawimage = cv2.imread(imagelist[i])
    rawimage = cv2.resize(rawimage,(X,Y))
    convertedarray[i] = rawimage

print(convertedarray.shape)

labels = LabelEncoder().fit_transform(labels)
labels = np_utils.to_categorical(labels)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(convertedarray, labels, test_size=0.4,random_state=10)

#####################################################

model = Sequential()

model.add(BatchNormalization(mode=1,input_shape=(Y,X,3)))

model.add(Convolution2D(32, 3, 3,border_mode='same',dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3,border_mode='same', dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3,border_mode='same', dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3,border_mode='same', dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.65))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.65))

model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(Xtrain,Ytrain,nb_epoch=15, batch_size=64,shuffle=True,callbacks=[early_stopping])




preds = model.predict(Xtest, verbose=1)

print("Validation Log Loss: {}".format(log_loss(Ytest, preds)))
submission = pd.DataFrame(preds, columns=classes)
submission.to_csv("cnn.csv")
