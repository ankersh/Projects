#I didn't end up using any of this in my final MLP model


from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

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



trainimages = []

for i in range(0, len(imagelist)):
    rawimage = cv2.imread(imagelist[i],0)
    rawimage = cv2.resize(rawimage,(154,87))
    timg = cv2.adaptiveThreshold(rawimage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,2)                #This or (9,4) was the most useful threshold I managed to find.
    image = timg.flatten()
    trainimages.append(image)



Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainimages, labels, test_size=0.4,random_state=10)          #Splits training data randomly




mlp = MLPClassifier(hidden_layer_sizes=(13398),max_iter=150,alpha=0.0001,verbose=True)


mlp.fit(Xtrain,Ytrain)


prediction = mlp.predict(Xtest)
loss = mlp.predict_proba(Xtest)

result = pd.DataFrame(prediction, columns=['pred'])
result.insert(0,'image',Ytest)
result.to_csv("ThreshClassPredictions.csv")

proba = pd.DataFrame(loss, columns=classes)
proba.to_csv("ThreshProb.csv")


print("Log Loss: " + str(log_loss(Ytest, prediction)))
