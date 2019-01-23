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
import io
data = pd.read_csv("./train.csv")
data['month'] = data['month'].map({'feb': 1, 'jan': 0, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11})
data['day'] = data['day'].map({'mon': 1, 'sun': 0, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6})
print data.corr(method='pearson')
used_features = [
	"DMC","temp","RH"
	]
features = data[used_features]
target = data["area"]

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.000001,max_iter=10000)
reg.fit(feature_train,target_train)
#print reg.score( feature_test,target_test )
pred = reg.predict(feature_test)
i=0
while i<len(pred):
	if pred[i]<0 :
		pred[i]=0
	i+=1
print r2_score(target_test,pred)

test = pd.read_csv("./test.csv")
test['month'] = test['month'].map({'feb': 1, 'jan': 0, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11})
test['day'] = test['day'].map({'mon': 1, 'sun': 0, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6})

test_fea = test[used_features]
aid = test["Id"]
res = reg.predict(test_fea)

i=0
while i<len(res):
	if res[i]<0 :
		res[i]=0
	i+=1

df = pd.DataFrame(data={"Id": aid, "Zarea": res})
df.to_csv("./file14.csv", sep=',',index=False)