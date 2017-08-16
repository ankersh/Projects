import cv2
import numpy as np
import os
import pandas as pd


train_dir = "sampleTrain"
classes = sorted(os.listdir(train_dir))[0:]

ProbeA = pd.read_csv('predictions.csv', names = classes)
ProbeA.to_csv("predictionss.csv")
