import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

data = pd.read_csv("./train.csv")

print data

print data.describe()

print data.head()

print data.shape

print data.dtypes

print data.corr(method='pearson')


used_features = ["MWG","NWG","KWG","MDIMC","NDIMC","MDIMA","NDIMB","KWI","VWM","VWN","STRM","STRN","SA","SB"]

features = data[used_features]
target1 = data["Run1"]
target2 = data["Run2"]
target3 = data["Run3"]
target4 = data["Run4"]

print features
print target1
print target2
print target3
print target4

#hist = data.hist()
#plt.show()

feature_train1, feature_test1, target_train1, target_test1 = train_test_split(features, target1, test_size=0.2, random_state=42)

feature_train2, feature_test2, target_train2, target_test2 = train_test_split(features, target2, test_size=0.2, random_state=42)

feature_train3, feature_test3, target_train3, target_test3 = train_test_split(features, target3, test_size=0.2, random_state=42)

feature_train4, feature_test4, target_train4, target_test4 = train_test_split(features, target4, test_size=0.2, random_state=42)

reg = ExtraTreesRegressor(n_estimators=500)
test = pd.read_csv("./test.csv")
test_fea = test[used_features]
aid = test["Id"]

reg.fit(features,target1)
y1=reg.predict(test_fea)

reg.fit(features,target2)
y2=reg.predict(test_fea)

reg.fit(features,target3)
y3=reg.predict(test_fea)

reg.fit(features,target4)
y4=reg.predict(test_fea)

df = pd.DataFrame(data={"Id": aid, "Run1 (ms)": y1,"Run2 (ms)": y2,"Run3 (ms)": y3,"Run4 (ms)": y4})
df.to_csv("./fileet3.csv", sep=',',index=False)
