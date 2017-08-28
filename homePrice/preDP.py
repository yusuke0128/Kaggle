#preDPクラス：データの前処理
import numpy as np 
import pandas as pd
import csv

class preDP:
	def _init_(self,csvName):
		self.csvName = csvName
	
	def setDF(self):
		dataFL = pd.read_csv(self.csvName,header=0)
	
	def getDF(self):
		return dataFL
	
	def mapDF(self): 
		#dataのダミー変数(途中まで)
		dataFL["MSZoning"] = dataFL["MSZoning"].map({"A":0,"C":1,"FV":2,"I":3,"RH":4,"RL":5,"RP":6,"RM":7}).astype(int)
		dataFL["Street"] = dataFL["Street"].map({"Grvl":0,"Pave":1}).astype(int)
		dataFL["Alley"] = dataFL["Alley"].map({"Grvl":0,"Pave":1,"NA":2}).astype(int)
		dataFL["LotShape"] = dataFL["LotShape"].map({"Reg":0,"IR1":1,"IR2":2,"IR3":3}).astype(int)
		dataFL["LandContour"] = dataFL["LandContour"].map({"Lvl":0,"Bnk":1,"HLS":2,"Low":3}).astype(int)
		dataFL["Utilities"] = dataFL["Utilities"].map({"ALLPub":0,"NoSewr":1,"NoSeWa":2,"ELO":3}).astype(int)
		dataFL["LotConfig"] = dataFL["LotConfig"].map({"Inside":0,"Coner":1,"CulDSac":2,"FR2":3,"FR3":4}).astype(int)
		dataFL["LandSlope"] = dataFL["LandSlope"].map({"Gtl":0,"Mod":1,"Sev":2}).astype(int)
		dataFL["Neighborhood"] = dataFL["Neighborhood"].map({"Blmngtn":0,"Blueste":1,"BrDale":2,"BrkSide":3}).astype(int)
	
	def supNA():