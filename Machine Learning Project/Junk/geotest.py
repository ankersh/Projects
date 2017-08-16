from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from skimage.feature import hog

import matplotlib
import matplotlib.pyplot as plt

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






inputc1 = np.float32([[0,72],[128,72],[128,0],[0,0]])
outputc1 = np.float32([[3,72],[128,68],[128,4],[1,4]])
t1 = cv2.getPerspectiveTransform(inputc1,outputc1)


inputc2 = np.float32([[0,72],[128,72],[128,0],[0,0]])
outputc2 = np.float32([[0,0],[128,1],[128,72],[1,72]])
t2 = cv2.getPerspectiveTransform(inputc2,outputc2)


img = cv2.imread(imagelist[666])
img = cv2.resize(img,(128,72))
dst = cv2.warpPerspective(img,t1,(128,72))


plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
