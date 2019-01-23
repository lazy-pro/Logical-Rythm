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
from sklearn.feature_selection import RFE
scaler = StandardScaler()
import io
data = pd.read_csv("./train.csv")
data['month'] = data['month'].map({'feb': 1, 'jan': 0, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11})
data['day'] = data['day'].map({'mon': 1, 'sun': 0, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6})

used_features = [
	"X",
    "Y",
    "month",
    "day","FFMC","DMC","DC","ISI","temp","RH","wind","rain"
	]
	
features = data[used_features]
target = data["area"]

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)


scaler.fit(feature_train)
feature_train = scaler.transform(feature_train)
feature_test = scaler.transform(feature_test)


from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(max_iter=10000,activation='tanh')
rfe = RFE(reg,4)
rfe.fit(feature_train,target_train)

print rfe.score(feature_test,target_test)

test = pd.read_csv("./test.csv")
test['month'] = test['month'].map({'feb': 1, 'jan': 0, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11})
test['day'] = test['day'].map({'mon': 1, 'sun': 0, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6})
scaler.fit(feature_train)
test_fea = test[used_features]
test_fea1 = scaler.transform(test_fea)
aid = test["Id"]
res = rfe.predict(test_fea1)

df = pd.DataFrame(data={"Id": aid, "Zarea": res})
df.to_csv("./file4.csv", sep=',',index=False)
