#This is just showing some of the geometric transformations I tried, I didn't end up using any of these for my final model.



from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from skimage.feature import hog



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
    rawimage = cv2.resize(rawimage,(128,72))
    h = hog(rawimage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    testimages.append(h)


images = []
for i in range(0, len(imagelist)):
    rawimage = cv2.imread(imagelist[i],0)
    rawimage = cv2.resize(rawimage,(128,72))
    images.append(rawimage)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(images, labels, test_size=0.3,random_state=10)          #Splits training data randomly

realtest = []                   #Gets HoG of the images in Xtest
for image in Xtest:
    h = hog(image,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    realtest.append(h)


inputc1 = np.float32([[0,72],[128,72],[128,0],[0,0]])               #Small perspective change of 1 to 2 pixels
outputc1 = np.float32([[2,72],[128,70],[127,2],[1,2]])
t1 = cv2.getPerspectiveTransform(inputc1,outputc1)

inputc3 = np.float32([[0,72],[128,72],[128,0],[0,0]])               #Larger perspective change of 4 to 5 pixels.
outputc3 = np.float32([[5,72],[128,67],[127,5],[5,1]])
t3 = cv2.getPerspectiveTransform(inputc3,outputc3)

inputc2 = np.float32([[0,72],[128,72],[128,0],[0,0]])               #Rotated the image by 180 degrees. I later used numpy to do this.
outputc2 = np.float32([[0,0],[128,0],[128,72],[0,72]])
t2 = cv2.getPerspectiveTransform(inputc2,outputc2)


transformedimages = []
transformedlabels = []

for i in range(0,(len(Xtrain)/3)):
    image = Xtrain[i]
    transformedlabels.append(Ytrain[i])
    timage = cv2.warpPerspective(image,t1,(128,72))
    h = hog(timage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    transformedimages.append(h)

for i in range((len(Xtrain)*2/4),(len(Xtrain))):
    image = Xtrain[i]
    transformedlabels.append(Ytrain[i])
    timage = cv2.warpPerspective(image,t2,(128,72))
    h = hog(timage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    transformedimages.append(h)


for i in range(0,(len(Xtrain)*3/4)):

    rimage = Xtrain[i]
    transformedlabels.append(Ytrain[i])
    h = hog(rimage,pixels_per_cell=(8, 8),cells_per_block=(3, 3))
    transformedimages.append(h)





mlp = MLPClassifier(hidden_layer_sizes=(7938),max_iter=150)


mlp.fit(transformedimages,transformedlabels)


prediction = mlp.predict_proba(testimages)


probs = pd.DataFrame(prediction, columns=classes)
probs.insert(0,'image',testname)
probs.to_csv("GeoTSubmission.csv")


loss = mlp.predict_proba(realtest)

cpred = mlp.predict(realtest)
result = pd.DataFrame(cpred, columns=['pred'])
result.insert(0,'image',Ytest)
result.to_csv("GEOPRED.csv")

print("Log Loss: " + str(log_loss(Ytest, prediction)))
