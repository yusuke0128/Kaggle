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
#データ読み込み
trainData = glob.glob("./train/*")
trainLabelDf = pd.read_csv("train_labels.csv")
for i in trainData:
	tmp = trainData
	number_padded = "%03d"%2
	print(number_padded)
	trainData[i] = tmp.find("./train/"+number_padded+".jpg")
print(trainData[0])


