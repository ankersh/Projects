from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

train_dir = "sampleTrain"
test_dir = "test"
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']



imagelist = []                  #Contains all the images
for c in classes:
    fish = [train_dir+'/'+c+'/'+item for item in os.listdir(train_dir+'/'+c)]
    imagelist.extend(fish)

labels = []                     #Contains the label for imagelist[i] at labels[i]
for c in classes:
    l = [c]*len(os.listdir(train_dir+'/'+c))
    labels.extend(l)

test = []
testname = []
for item in os.listdir(test_dir+'/'):
    test.append(test_dir+'/'+item)
    testname.append(item)


print len(imagelist)
print len(labels)

trainimages = []

for i in range(0, len(imagelist)):
    rawimage = cv2.imread(imagelist[i],0)
    rawimage = cv2.resize(rawimage,(128,72))
    h = hog(rawimage)
    trainimages.append(h)

testimages = []
for i in range(0, len(test)):
    rawimage = cv2.imread(test[i],0)
    rawimage = cv2.resize(rawimage,(128,72))
    h = hog(rawimage)
    testimages.append(h)



Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainimages, labels, test_size=0.4,random_state=10)          #Splits training data randomly


mlp = MLPClassifier(hidden_layer_sizes=(len(Xtrain[2])))


mlp.fit(Xtrain,Ytrain)


prediction = mlp.predict_proba(Xtest)
testpreds = mlp.predict_proba(testimages)
print("Validation Log Loss: {}".format(log_loss(Ytest, prediction)))

result = pd.DataFrame(prediction, columns=classes)
result.to_csv("HOGresult.csv")


submission = pd.DataFrame(testpreds, columns=classes)
submission.insert(0,'image',testname)
submission.to_csv("submissionHOG.csv")
