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

used_features = ["MWG","NWG","KWG","MDIMC","NDIMC","MDIMA","NDIMB","KWI","VWM","VWN","STRM","STRN","SA","SB"]

features = data[used_features]
target1 = data["Run1"]
target2 = data["Run2"]
target3 = data["Run3"]
target4 = data["Run4"]

print features

print target1

sLength = len(features["MWG"])
features = features.assign(e=pd.Series(target1).values)
print features