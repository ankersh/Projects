from sklearn.neural_network import MLPClassifier
import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


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
    l = [c for i in os.listdir(train_dir+'/'+c)]
    labels.extend(l)

test = []
testname = []
for item in os.listdir(test_dir+'/'):
    test.append(test_dir+'/'+item)
    testname.append(item)


#inputc2 = np.float32([[0,87],[154,87],[154,0],[0,0]])
#outputc2 = np.float32([[0,28],[154,28],[154,100],[0,100]])
#t2 = cv2.getPerspectiveTransform(inputc2,outputc2)


inputc1 = np.float32([[0,87],[154,87],[154,0],[0,0]])                                   #Adds borders on top and bottom so image is 154 * 154
outputc1 = np.float32([[0,100],[154,100],[154,28],[0,28]])
t1 = cv2.getPerspectiveTransform(inputc1,outputc1)


ict1 = np.float32([[0,87],[154,87],[154,0],[0,0]])                                   #Adds borders on top and bottom so image is 154 * 154
oct1 = np.float32([[0,100],[154,100],[154,28],[0,28]])
croptop = cv2.getPerspectiveTransform(ict1,oct1)

ics1 = np.float32([[55,0],[128,0],[36,154],[118,154]])                                   #Adds borders on top and bottom so image is 154 * 154
ocs1 = np.float32([[0,100],[154,100],[154,28],[0,28]])
cropsides = cv2.getPerspectiveTransform(ics1,ocs1)



M1 = cv2.getRotationMatrix2D((154/2,154/2),90,1)
M2 = cv2.getRotationMatrix2D((154/2,154/2),180,1)
M3 = cv2.getRotationMatrix2D((154/2,154/2),270,1)


# 991
# 632


#img = cv2.imread(imagelist[991],0)
#img = cv2.resize(img,(140,79))

#ret,timg = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#timg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,4)




pred = pd.read_csv("RHOG2Submission.csv")
ar = pred.as_matrix(columns=classes)
c = pred[['image']]
ar = np.clip(ar,0.001,0.99)

pred2 = pd.DataFrame(ar,columns=classes)
pred2.insert(0,'image',c)
pred2.to_csv("SubRHOG2.csv")




#img = cv2.warpPerspective(img,t1,(154,154))


#timg = cv2.warpPerspective(img,t2,(154,154))

#timg = cv2.warpAffine(img,M1,(154,154))





plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(timg),plt.title('Output')
plt.show()





#timg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,7,2)
#img = cv2.GaussianBlur(img,(1,1),0)
