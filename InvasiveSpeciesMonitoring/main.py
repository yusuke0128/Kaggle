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

gpu_flag = 0

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

p  = pre.PreProc()
#データ読み込み
trainDataURL = "./dataset/train/*"
trainLabelDfURL = "train_labels.csv"
print(trainDataURL)
trainLabelSet = p.preLabel(trainLabelDfURL)
trainRGBSet = p.dataToRGB(trainDataURL)
