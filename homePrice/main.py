import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from sklearn import linear_model as Lm

#trainデータ読み込み
trainDataFL = pd.read_csv("train.csv",header=0)
print(trainDataFL)
#土地面積と地価の関係をプロット
plt.plot(trainDataFL["LotArea"],trainDataFL["SalePrice"],"r.")
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
#RANSACによる回帰直線推定
ransac = Lm.RANSACRegressor()
x = (trainDataFL["LotArea"].values).reshape(len(trainDataFL.index),1)
y = (trainDataFL["SalePrice"].values).reshape(len(trainDataFL.index),1)
ransac.fit(x,y)
yRansac = ransac.predict(x)
plt.plot(x,yRansac,"b-")
plt.show()
