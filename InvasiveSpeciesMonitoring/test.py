#coding:utf-8
import chainer
import numpy as np
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers as opt
import time
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import cv2
import preProc as pre
import _pickle as cPickle
gpu_flag = 0

if gpu_flag >= 0:
	cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 10
p  = pre.PreProc()

#学習データ読み込み
testDataURL = "./dataset/test/*"
testLabelDfURL = "sample_submission.csv"
testLabelSet = p.preLabel(testLabelDfURL).astype(np.int32)
testRGBSet = p.dataToRGB(testDataURL).astype(np.float32)
N = testRGBSet.shape[0]

#モデル読み込み
model = cPickle.load(open("model.pkl", "rb"))

def forward(xData, yData, train=False):
	x,t = chainer.Variable(xData),chainer.Variable(yData)
	h = model.conv1(x)
	h = F.max_pooling_2d(model.b1(F.relu(model.conv2(h))),2)
	h = F.max_pooling_2d(model.b2(F.relu(model.conv3(h))),2)
	h = model.conv4(h)
	h = model.conv5(h)
	h = F.dropout(model.b3(F.relu(model.l1(h))),train=train)
	h = F.dropout(model.b4(F.relu(model.l2(h))),train=train)
	h = F.dropout(model.b5(F.relu(model.l3(h))),train=train)
	y = model.l4(h)
	if train:
		return F.softmax_cross_entropy(y,t)
	else:
		print(y.data)
		return y

for i in range(0,N,batchsize):
	xBatch = testRGBSet[i:i+batchsize].astype(np.float32)
	yBatch = testLabelSet[i:i+batchsize].astype(np.int32)
	a  = forward(xBatch,yBatch, train=False)
	if gpu_flag>=0:
		xBatch = cuda.to_gpu(xBatch)
		yBatch = cuda.to_gpu(yBatch)


