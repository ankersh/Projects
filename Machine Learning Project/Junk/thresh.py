from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv
from skimage.feature import daisy
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
    rawimage = cv2.resize(rawimage,(154,87))
    timg = cv2.adaptiveThreshold(rawimage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,2)
    image = timg.flatten()
    trainimages.append(image)


print len(trainimages[5])

#testimages = []
#for i in range(0, len(test)):
#    rawimage = cv2.imread(test[i],0)
#    rawimage = cv2.resize(rawimage,(128,72))
#    h = hog(rawimage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
#    testimages.append(h)



Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainimages, labels, test_size=0.4,random_state=10)          #Splits training data randomly




mlp = MLPClassifier(hidden_layer_sizes=(13398),max_iter=150,alpha=0.0001,verbose=True)


mlp.fit(Xtrain,Ytrain)


prediction = mlp.predict(Xtest)
loss = mlp.predict_proba(Xtest)
print("Validation Log Loss: {}".format(log_loss(Ytest, loss)))

result = pd.DataFrame(prediction, columns=['pred'])
result.insert(0,'image',Ytest)
result.to_csv("ThreshClassPredictions.csv")

proba = pd.DataFrame(loss, columns=classes)
proba.to_csv("ThreshProb.csv")

#testpreds = mlp.predict_proba(testimages)
#submission = pd.DataFrame(testpreds, columns=classes)
#submission.insert(0,'image',testname)
#submission.to_csv("submissionDAISY.csv")
