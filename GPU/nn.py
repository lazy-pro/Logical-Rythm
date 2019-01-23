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

scaler = StandardScaler()
data = pd.read_csv("./train.csv")

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

feature_train1, feature_test1, target_train1, target_test1 = train_test_split(features, target1, test_size=0.2, random_state=42)

feature_train2, feature_test2, target_train2, target_test2 = train_test_split(features, target2, test_size=0.2, random_state=42)

feature_train3, feature_test3, target_train3, target_test3 = train_test_split(features, target3, test_size=0.2, random_state=42)

feature_train4, feature_test4, target_train4, target_test4 = train_test_split(features, target4, test_size=0.2, random_state=42)

scaler.fit(feature_train1)
feature_train1 = scaler.transform(feature_train1)
feature_test1 = scaler.transform(feature_test1)

scaler.fit(feature_train2)
feature_train2 = scaler.transform(feature_train2)
feature_test2 = scaler.transform(feature_test2)

scaler.fit(feature_train3)
feature_train3 = scaler.transform(feature_train3)
feature_test3 = scaler.transform(feature_test3)

scaler.fit(feature_train4)
feature_train4 = scaler.transform(feature_train4)
feature_test4 = scaler.transform(feature_test4)


print "scaling done"
print feature_train1
print feature_train2
print feature_train3
print feature_train4

print feature_test1
print feature_test2
print feature_test3
print feature_test4

from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(max_iter=500)

test = pd.read_csv("./test.csv")
test_fea = test[used_features]
aid = test["Id"]

#RUN1
reg.fit(feature_train1,target_train1)
print "RUN1",reg.score(feature_test1,target_test1)


#RUN2
reg.fit(feature_train2,target_train2)
print "RUN2",reg.score(feature_test2,target_test2)


#RUN3
reg.fit(feature_train3,target_train3)
print "RUN3",reg.score(feature_test3,target_test3)


#RUN4
reg.fit(feature_train4,target_train4)
print "RUN4",reg.score(feature_test4,target_test4)
