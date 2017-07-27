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

batchsize = 135
n_epoch = 200
p  = pre.PreProc()

#学習データ読み込み
trainDataURL = "./dataset/train/*"
trainLabelDfURL = "train_labels.csv"
trainLabelSet = p.preLabel(trainLabelDfURL)
trainRGBSet = p.dataToRGB(trainDataURL)
N = trainRGBSet.shape[0]
#model定義

model = chainer.FunctionSet(conv1 = L.Convolution2D(3,30,5),
                            conv2 = L.Convolution2D(30,30,11),
                            conv3 = L.Convolution2D(30,20,11),
                            conv4 = L.Convolution2D(20,18,11),
                            conv5 = L.Convolution2D(18,15,11),
                            l1 = L.Linear(6525,1000),
                            l2 = L.Linear(1000,500),
                            l3 = L.Linear(500,20),
                            l4 = L.Linear(20,2),
                            b1 = L.BatchNormalization(30),
                            b2 = L.BatchNormalization(20),
                            b3 = L.BatchNormalization(1000),
                            b4 = L.BatchNormalization(500),
                            b5 = L.BatchNormalization(20))

if gpu_flag >= 0:
	cuda.get_device(gpu_flag).use()
	model.to_gpu()

def forward(xData, yData, train=True):
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
		return F.accuracy(y,t)

optimizer = opt.Adam()
optimizer.setup(model)

fp2 = open("loss.txt", "w")
fp2.write("epoch\ttrain_loss\n")


#訓練ループ
startTime = time.clock()
for epoch in range(1,n_epoch+1):
	print("epoch%d"%epoch)
	perm = np.random.permutation(N)
	sumLoss = 0
	for i in range(0,N,batchsize):
		xBatch = trainRGBSet[perm[i:i+batchsize]].astype(np.float32)
		yBatch = trainLabelSet[perm[i:i+batchsize]].astype(np.int32)
		if gpu_flag>=0:
			xBatch = cuda.to_gpu(xBatch)
			yBatch = cuda.to_gpu(yBatch)

		optimizer.zero_grads()
		loss = forward(xBatch,yBatch)
		loss.backward()
		optimizer.update()
		sumLoss += float(cuda.to_cpu(loss.data)) * batchsize
	print("train mean loss: %f" % (sumLoss / N))
	fp2.write("%d\t%f\n" % (epoch, sumLoss / N))
	fp2.flush()

endTime = time.clock()
print(endTime - startTime)
# 学習モデル保存
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)
