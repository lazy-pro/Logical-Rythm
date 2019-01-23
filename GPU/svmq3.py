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
used_features = ["MWG","NWG","KWG","MDIMC","NDIMC","MDIMA","NDIMB","KWI","VWM","VWN","STRM","STRN","SA","SB"]

features = data[used_features]
target1 = data["Run1"]
target2 = data["Run2"]
target3 = data["Run3"]
target4 = data["Run4"]

feature_train1, feature_test1, target_train1, target_test1 = train_test_split(features, target1, test_size=0.2, random_state=42)

feature_train2, feature_test2, target_train2, target_test2 = train_test_split(features, target2, test_size=0.2, random_state=42)

feature_train3, feature_test3, target_train3, target_test3 = train_test_split(features, target3, test_size=0.2, random_state=42)

feature_train4, feature_test4, target_train4, target_test4 = train_test_split(features, target4, test_size=0.2, random_state=42)

from sklearn.svm import SVR
reg = SVR(C=1.0, epsilon=0.2)

reg.fit(feature_train1,target_train1)
print "RUN1", reg.score(feature_test1,target_test1)
reg.fit(feature_train2,target_train2)
print "RUN2", reg.score(feature_test2,target_test2)
reg.fit(feature_train3,target_train3)
print "RUN3", reg.score(feature_test3,target_test3)
reg.fit(feature_train4,target_train4)
print "RUN4", reg.score(feature_test4,target_test4)