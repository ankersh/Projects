from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

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
    rawimage = cv2.resize(rawimage,(144,77))
    h = hog(rawimage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    trainimages.append(h)




testimages = []
for i in range(0, len(test)):
    rawimage = cv2.imread(test[i],0)
    rawimage = cv2.resize(rawimage,(144,77))
    h = hog(rawimage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    testimages.append(h)



Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainimages, labels, test_size=0.25,random_state=10)          #Splits training data randomly



mlp = MLPClassifier(hidden_layer_sizes=(11088),max_iter=100,alpha=0.001,learning_rate_init=0.00005,verbose=True)


mlp.fit(Xtrain,Ytrain)


prediction = mlp.predict(Xtest)
loss = mlp.predict_proba(Xtest)
testpreds = mlp.predict_proba(testimages)

result = pd.DataFrame(prediction, columns=['pred'])
result.insert(0,'image',Ytest)
result.to_csv("HOGClassPredictions.csv")


submission = pd.DataFrame(testpreds, columns=classes)
submission.insert(0,'image',testname)
submission.to_csv("submissionHOG2.csv")

print("Log Loss: " + str(log_loss(Ytest, prediction)))
