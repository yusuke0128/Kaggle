#coding: utf-8
import _pickle as cPickle 
import matplotlib.pyplot as plt
import numpy as np
import chainer
import chainer.functions as F
model = cPickle.load(open("model.pkl", "rb"))

# 1つめのConvolution層の重みを可視化
print(model.conv1.W.shape)

n1,n2,h,w = model.conv1.W.shape  # modelは学習済みモデル
print(n1,n2,h,w)
img = F.transpose(model.conv1.W[0],(1,2,0)) # (h, w, channel)に変換
plt.imshow(img)
