from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv

from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from skimage.feature import peak_local_max
from skimage.morphology import watershed

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


testimage = cv2.imread(imagelist[0])
testimage = cv2.resize(testimage,(128,72))
shifted = cv2.pyrMeanShiftFiltering(testimage, 21, 51)
gshifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gshifted, 0, 255, cv2.THRESH_OTSU)[1]

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
