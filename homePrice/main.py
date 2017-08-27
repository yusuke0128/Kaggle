import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from sklearn import linear_model as Lm

#trainデータ読み込み
trainDataFL = pd.read_csv("train.csv",header=0)
print(trainDataFL)
#土地面積と地価の関係をプロット
plt.hold(True)
plt.plot(trainDataFL["LotArea"],trainDataFL["SalePrice"],"r.")
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
#RANSACによる回帰直線推定
ransac = Lm.RANSACRegressor(residual_threshold=20,max_trials=1000,stop_n_inliers=300)
x = (trainDataFL["LotArea"].values).reshape(len(trainDataFL.index),1)
y = (trainDataFL["SalePrice"].values).reshape(len(trainDataFL.index),1)
ransac.fit(x,y)
yRansac = ransac.predict(x)
a = ransac.estimator_.coef_
b = ransac.estimator_.intercept_
plt.plot(x,a*x+b,"b-")
plt.show()
print(ransac.estimator_.coef_)
print(ransac.estimator_.intercept_)
plt.hold(False)