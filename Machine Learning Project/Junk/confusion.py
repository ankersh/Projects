import os
import cv2
import pandas as pd
import numpy as np
import csv

from sklearn.metrics import confusion_matrix

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


pred = pd.read_csv("RHOGPRED.csv")

real = pred.as_matrix(columns=['image'])
guess = pred.as_matrix(columns=['pred'])

print confusion_matrix(real, guess, labels=classes)
