from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv
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



trainimages = []

for i in range(0, len(imagelist)):
    rawimage = cv2.imread(imagelist[i],0)
    rawimage = cv2.resize(rawimage,(144,77))
    rawimage = rawimage.flatten()
    trainimages.append(rawimage)

testimages = []
for i in range(0, len(test)):
    rawimage = cv2.imread(test[i],0)
    rawimage = cv2.resize(rawimage,(144,77))
    rawimage = rawimage.flatten()
    testimages.append(rawimage)




Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainimages, labels, test_size=0.25,random_state=10)          #Splits training data randomly


mlp = MLPClassifier(hidden_layer_sizes=(11088),learning_rate_init=0.001,alpha=0.0001,max_iter=300,verbose=True,tol=0.0001)


mlp.fit(Xtrain,Ytrain)


prediction = mlp.predict_proba(Xtest)
testpreds = mlp.predict_proba(testimages)

result = pd.DataFrame(prediction, columns=classes)
result.to_csv("MLPresult.csv")

cpred = mlp.predict(Xtest)
result = pd.DataFrame(cpred, columns=['pred'])
result.insert(0,'image',Ytest)
result.to_csv("MLPPRED.csv")

submission = pd.DataFrame(testpreds, columns=classes)
submission.insert(0,'image',testname)
submission.to_csv("submissionMLP.csv")

print("Log Loss: " + str(log_loss(Ytest, prediction)))
