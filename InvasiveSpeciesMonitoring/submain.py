#coding:utf-8
import chainer
import numpy as np
from chainer import cuda
import chainer.function as F
from chainer import optimizers as opt
import time
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import cv2

gpu_flag = 0

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

#辞書順から自然順に並び替えるメソッド
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#データ読み込み
trainDataSet = sorted(glob.glob("./dataset/train/*"),key=numericalSort)
trainLabelDf = pd.read_csv("train_labels.csv") 
trainR = 0
trainRGBSet = [0]*len(trainDataSet)
trainRSet = [0]*len(trainDataSet)
for i in range(len(trainDataSet)):
	trainDataSet[i] = cv2.imread(trainDataSet[i])
	trainRGBSet[i] = cv2.split(trainDataSet[i])
	trainRGB = trainRGBSet[i]
	trainRSet[i] = trainRGB[1]
	trainGSet[i] = trainRGB[2]
	trainBSet[i] = trainRGB[3]
cv2.imshow("orignal",trainRSet[1])
cv2.waitKey()
cv2.destroyAllWindows()
trainData = trainData.astype(xp.float32)
trainLabel = trainLabelDf["invasive"].value
