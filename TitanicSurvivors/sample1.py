# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
#学習データ読み込み
trainDataDf = pd.read_csv("train.csv",header=0)
#ダミー変数変換
trainDataDf["Gender"] = trainDataDf["Sex"].map({"female":0,"male":1}).astype(int)
meanAge = trainDataDf["Age"].dropna().median()
trainDataDf["Age"] = trainDataDf["Age"].fillna(meanAge)
trainDataDf = trainDataDf.drop(["Name","Ticket","Sex","SibSp","Parch","Fare","Cabin","Embarked","PassengerId"],axis=1)

#テストデータ読み込み
testDataDf = pd.read_csv("test.csv",header=0)
#ダミー変数変換
testDataDf["Gender"] = testDataDf["Sex"].map({"female":0,"male":1}).astype(int)
meanAge = testDataDf["Age"].dropna().median()
testDataDf["Age"] = testDataDf["Age"].fillna(meanAge)
ids = testDataDf["PassengerId"].values
testDataDf = testDataDf.drop(["Name","Ticket","Sex","SibSp","Parch","Fare","Cabin","Embarked","PassengerId"],axis=1)
#予測フェイズ
trainData = trainDataDf.values
testData = testDataDf.values
model = RandomForestClassifier(n_estimators=100)
output = model.fit(trainData[0::,1::],trainData[0::,0]).predict(testData).astype(int)

#結果を"titanic_submit.csv"として書き出す
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId","Survived"])
file_object.writerows(zip(ids, output))
submit_file.close() 
