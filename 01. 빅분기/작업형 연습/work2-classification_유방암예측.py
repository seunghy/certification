# %%
## 1. import lib and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegressionCV

train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/test.csv")

# %%
## 2. na/null값 확인
Y = train['diagnosis']
Y = Y.map(lambda x: 0 if x == 'B' else 1)

X = train.drop(columns=['diagnosis'])

print(Y.isna().sum().sum(), Y.isnull().sum().sum())
print(X.isna().sum().sum(), X.isnull().sum().sum())

# %%
## 3. 불필요한 col 제거
X.columns
X.drop(columns=['id'], inplace=True)
X_test = test.drop(columns=['id'])

# %%
## 4. 변수들 전처리
X.info()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# %%
## 5. Dummy
# PASS

# %%
## 6. train test split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, stratify=Y)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

# %%
## 7. model fit and check
rf = RandomForestClassifier(random_state=2022)
rf.fit(x_train, y_train)

rf_pred = rf.predict(x_val)
rf_pred_prob = rf.predict_proba(x_val)

print("roc_auc: ", roc_auc_score(y_val, rf_pred_prob[:,1]))
print("acc: ", rf.score(x_val, y_val))
print("f1: ", f1_score(y_val, rf_pred))
print("f1_cross val: ", cross_val_score(rf, x_val, y_val, scoring='f1').mean())



# %%
## 8. re-fit and submit
rf = RandomForestClassifier(random_state=2022)
rf.fit(X, Y)

rf_pred = rf.predict(X_test)

pd.DataFrame({'id':test['id'], 'pred':rf_pred})


# %%
