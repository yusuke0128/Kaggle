#coding:utf-8
import chainer
import numpy as np
from chainer import cuda
import chainer.function as F
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

batchsize = 200
n_epoch = 500
p  = pre.PreProc()

#学習データ読み込み
trainDataURL = "./dataset/train/*"
trainLabelDfURL = "train_labels.csv"
trainLabelSet = p.preLabel(trainLabelDfURL)
trainRGBSet = p.dataToRGB(trainDataURL)

N = trainRGBSet.shape[0]
#model定義

model = chainer.FunctionSet(conv1 = L.Convolution2D(3,100,5),
                            conv2 = L.Convolution2D(100,200,51),
                            conv3 = L.Convolution2D(200,250,11),
                            conv4 = L.Convolution2D(250,200,11),
                            conv5 = L.Convolution2D(200,150,23),
                            conv6 = L.Convolution2D(150,120,11),
                            conv7 = L.Convolution2D(120,100,31),
                            conv8 = L.Convolution2D(100,60,51),
                            conv9 = L.Convolution2D(60,20,11), 
                            l1 = L.Linear(11440,5000),
                            l2 = L.Linear(5000,2000),
                            l3 = L.Linear(2000,100),
                            l4 = L.Linear(100,2))

if gpu_flag >= 0:
	cuda.get_device(gpu_flag).use()
	model.to_gpu()

def forward(xData, yData, train=True):
	x,t = chainer.Variable(xData),chainer.Variable(yData)
	h = model.conv1(x)
	h = F.max_pooling_2d(F.relu(model.conv2(h)),2)
	h = model.conv3(h)
	h = model.conv4(h)
	h = F.max_pooling_2d(F.relu(model.conv5(h)),2)
	h = model.conv6(h)
	h = F.max_pooling_2d(F.relu(model.conv7(h)),2)
	h = model.conv8(h)
	h = model.conv9(h)
	h = F.dropout(F.relu(model.l1(h)),train=train)
	h = F.dropout(F.relu(model.l2(h)),train=train)
	h = F.dropout(F.relu(model.l3(h)),train=train)	
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
		xBatch = trainRGBSet[perm[i:i+batchsize]].astype(float32)
		yBatch = trainLabelSet[perm[i:i+batchsize]].astype(float32)
		if gpu_flag>=0:
			xBatch = cuda.to_gpu(xBatch)
			yBatch = cuda.to_gpu(yBatch)

		optimizer.zero_grads()
		loss,acc = forward(xBatch,yBatch)
		loss.backward()
		optimizer.update()
		sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
		sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
	print("train mean loss: %f" % (sum_loss / N))
	fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
	fp2.flush()

end_time = time.clock()
print(end_time - start_time)
# 学習モデル保存
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)
