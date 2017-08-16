from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from skimage.feature import hog

import matplotlib
import matplotlib.pyplot as plt

traind = "/sampleTrain"
testd = "/test"
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


imagelist = []                  #Contains all the raw images
for c in classes:
    fish = [traind+'/'+c+'/'+item for item in os.listdir(traind+'/'+c)]
    imagelist.extend(fish)

labels = []                     #Contains the label for imagelist[i] at labels[i]
for c in classes:
    l = [c for i in os.listdir(traind+'/'+c)]
    labels.extend(l)

test = []
testname = []
for item in os.listdir(test_dir+'/'):
    test.append(test_dir+'/'+item)
    testname.append(item)


testimages = []
for i in range(0, len(test)):
    rawimage = cv2.imread(test[i],0)
    rawimage = cv2.resize(rawimage,(154,87))
    h = hog(rawimage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    testimages.append(h)


images = []
for i in range(0, len(imagelist)):
    rawimage = cv2.imread(imagelist[i],0)
    rawimage = cv2.resize(rawimage,(154,87))
    images.append(rawimage)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(images, labels, test_size=0.35,random_state=10)          #Splits training data randomly

realtest = []
for image in Xtest:
    h = hog(image,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    realtest.append(h)

print len(realtest[0])


transformedimages = []
transformedlabels = []




for i in range(0,30):
     ran = random.random()
     if ran <= 0.25:
         image = Xtrain[i]                                                          #Randomly adds in 1/4 of the training images unrotated
         transformedlabels.append(Ytrain[i])
         h = hog(image,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
         transformedimages.append(h)


for i in range(0,(len(Xtrain)/3)):                                                     #Rotates the first third of images by 90 degrees
    image = Xtrain[i]
    transformedlabels.append(Ytrain[i])
    timage = np.rot90(image,1)
    h = hog(timage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    transformedimages.append(h)


for i in range((len(Xtrain)/3),(2*len(Xtrain)/3)):                                     #Rotates the second third of images by 180 degrees
    image = Xtrain[i]
    transformedlabels.append(Ytrain[i])
    timage = np.rot90(image,2)
    h = hog(timage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    transformedimages.append(h)

for i in range((2*len(Xtrain)/3),len(Xtrain)):                                          #Rotates the last third by 270 degrees
    image = Xtrain[i]
    transformedlabels.append(Ytrain[i])
    timage = np.rot90(image,3)
    h = hog(timage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    transformedimages.append(h)





mlp = MLPClassifier(hidden_layer_sizes=(11016),max_iter=150,verbose=True,tol=0.0001)


mlp.fit(transformedimages,transformedlabels)


prediction = mlp.predict_proba(testimages)


probs = pd.DataFrame(prediction, columns=classes)
probs.insert(0,'image',testname)
probs.to_csv("RHOG2Submission.csv")


loss = mlp.predict_proba(realtest)

cpred = mlp.predict(realtest)
result = pd.DataFrame(cpred, columns=['pred'])
result.insert(0,'image',Ytest)
result.to_csv("RHOG2PRED.csv")

print("Log Loss: " + str(log_loss(Ytest, prediction)))
