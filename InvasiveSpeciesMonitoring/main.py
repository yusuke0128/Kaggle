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
import preProc as pre
import cPickle
gpu_flag = 0

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 100
n_epoch = 20
p  = pre.PreProc()

#学習データ読み込み
trainDataURL = "./dataset/train/*"
trainLabelDfURL = "train_labels.csv"
trainLabelSet = p.preLabel(trainLabelDfURL)
trainRGBSet = p.dataToRGB(trainDataURL)
#テストデータ読み込み
testDataURL = "./dataset/test/*"
testRGBSet = p.dataToRGB(testDataURL)
#model定義

model = chainer.FunctionSet(conv1 = F.Convolution2D(3,100,5),
                            conv2 = F.Convolution2D(100,200,21),
                            conv3 = F.Convolution2D(200,250,51),
                            conv4 = F.Convolution2D(250,200,11),
                            conv5 = F.Convolution2D(200,170,51),
                            conv6 = F.Convolution2D(170,100,21),
                            conv7 = F.Convolution2D()
                            conv8 = 
                            l1=F.liner()
                            l2=F.liner()
                            l3= )

if gpu_flag >= 0:
	cuda.get_device(gpu_flag).use()
	model.to_gpu()

def forward():
	x,t = chainer.Variable(trainRGBSet),chainer.Variable(trainLabelSet)
	h = model.conv1(x)
	h = model.conv2(h)
	h = F.max_pooling_2d(F.relu(model.conv3(h)),10)
	h = F.max_pooling_2d(F.relu(model.conv3(h)),10)
	h = F.dropout(F.relu(model.l1(h)),train=train)
	h = F.dropout(F.relu(model.l2(h)),train=train)
	y = model.l3(h)
	if train:
		return F.softmax_cross_entropy(y,t)
	else:
		return F.accuracy(y,t)

optimizer = optimizers.Adam()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")

fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")


#訓練ループ
startTime = time.clock()
for epoch in range(1,n_epoch+1)
	print("epoch%d"%epoch)
	perm = np.random.permutation(N)
	sumLoss = 0
	for i in xrange(0,N,batchsize):
		xBatch = trainRGBSet[perm[i:i+batchsize]]
		yBatch = trainLabelSet[perm[i:i+batchsize]]
		if gpu_flag>0
			xBatch = cuda.to_gpu(xBatch)
			yBatch = cuda.to_gpu(yBatch)

		optimizer.zero_grads()
		loss,acc = forward(xBatch,yBatch)
		optimaizer.update()
        	sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        	sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
	print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N) 

# 学習モデル保存
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)
