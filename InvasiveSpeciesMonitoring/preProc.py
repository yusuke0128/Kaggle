import glob
import pandas as pd
import re
import sort
import cv2
import chainer
import numpy as np
from chainer import cuda

class PreProc():
	def __init__(self):
		self.URL = ""
		self.value=0

	def dataToRGB(self,URL):
		s = sort.Sort()
		dataSet = sorted(glob.glob(URL),key=s.numericalSort)
		RGBSet = [0]*len(dataSet)
		for i in range(len(dataSet)):
			dataSet[i] = cv2.imread(dataSet[i])
			dataSet[i] = cv2.resize(dataSet[i],(230,174))
			RGBSet[i] = cv2.split(dataSet[i])
		RGBSet = np.array(RGBSet)
		print(RGBSet.shape)
		return RGBSet

	def preLabel(self,URL):
		labelDf = pd.read_csv(URL)
		label = labelDf["invasive"].values
		label = np.array(label)
		return label

