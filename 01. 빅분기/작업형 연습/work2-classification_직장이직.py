
# %%
### 1. import library and data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor

# %%
X = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_train.csv")
Y = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_test.csv")


# %%
### 2. 불필요한 cols 제거
X.drop(columns=['enrollee_id'], inplace=True)
X_test.drop(columns=['enrollee_id'], inplace=True)


# %%
### 3. check na/null값
print("X null/na: ", X.isna().sum(), X.isnull().sum())
print("Y null/na: ", Y.isna().sum(), Y.isnull().sum())
print("X_test null/na: ", X_test.isna().sum(), X_test.isnull().sum())

# %%
# 수치형 na값 -> 평균으로 대체, 명목형 null값 -> 최빈값으로 대체
col1 = X.select_dtypes(exclude='object').columns
col2 = X.select_dtypes(include='object').columns

for i in range(len(col1)):
    rows = X.loc[:,col1[i]].isna()
    X.loc[rows, col1[i]] = X.loc[:,col1[i]].mean()
    
    rows = X_test.loc[:,col1[i]].isna()
    X_test.loc[rows, col1[i]] = X_test.loc[:,col1[i]].mean()

for j in range(len(col2)):
    rows = X.loc[:,col2[j]].isna()
    X.loc[rows, col2[j]] = X.loc[:,col2[j]].mode()[0]
    
    rows = X_test.loc[:,col2[j]].isna()
    X_test.loc[rows, col2[j]] = X_test.loc[:,col2[j]].mode()[0]
    
print("X null/na: ", X.isna().sum().sum(), X.isnull().sum().sum())
print("Y null/na: ", Y.isna().sum().sum(), Y.isnull().sum().sum())
print("X_test null/na: ", X_test.isna().sum().sum(), X_test.isnull().sum().sum())

    
# %%
### 4. 변수값 확인 및 전처리
X.describe()

# training_hours 변수가 너무 극단값이 많아서 np.log 변환시킴
X['training_hours'] = np.log(X['training_hours'])
X_test['training_hours'] = np.log(X_test['training_hours'])

# %%
### 5. category & scaling
for i in range(len(col2)):
    print(col2[i],"::", X[col2[i]].unique())

# dummy 추가 ver.
Dummy = False
encoder = LabelEncoder()
for j in range(len(col2)):
    X['train'] = 1
    X_test['train'] = 0
    tmp = pd.concat([X, X_test], axis=0)
    encoder.fit(tmp[col2[j]])
    tmp[col2[j]] = encoder.transform(tmp[col2[j]])
    
    if Dummy:
        tmp = pd.concat([tmp, pd.get_dummies(tmp[col2[j]], prefix=col2[j])], axis=1)
        
    X = tmp.loc[tmp['train'] == 1,:]
    X_test = tmp.loc[tmp['train'] == 0,:]
        
if Dummy:
    X.drop(columns=col2, inplace=True)
    X_test.drop(columns=col2, inplace=True)    

# 수치형 variable
scaler = MinMaxScaler()
X[col1] = scaler.fit_transform(X[col1])
X_test[col1] = scaler.transform(X_test[col1])

# display(X.head(2))
print(X.shape, X_test.shape)

# %%
### 6. train test split
x_train, x_val, y_train, y_val = train_test_split(X, Y.target, test_size=0.3, stratify=Y.target)

print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)

# %%
### 7. model fit
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=2022)
rf.fit(x_train, y_train)
rf_pred_proba = rf.predict_proba(x_val)

print("rf_acc: ", rf.score(x_val, y_val))
print("rf_AUC: ", roc_auc_score(y_val, rf_pred_proba[:,1]))
print("cross_val: ", cross_val_score(rf, x_val, y_val, cv=5).mean())


# %%
# gridsearch
params = {'n_estimators':[100,150,200,250,300], 'max_depth':[3,5,7,9]}

grid = GridSearchCV(rf, param_grid=params, cv=5)
grid.fit(x_train, y_train)
print("최적 params: ", grid.best_params_)

best_params = grid.best_estimator_
rf_pred_proba_best = best_params.predict_proba(x_val)

print("rf_acc_best: ", best_params.score(x_val, y_val))
print("rf_AUC_best: ", roc_auc_score(y_val, rf_pred_proba_best[:,1]))
print("cross_val_best: ", cross_val_score(best_params, x_val, y_val, cv=5).mean())

# %%
# XGB
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
xgb_pred_proba = xgb.predict_proba(x_val)[:,1]

print("xgb_acc: ", xgb.score(x_val, y_val))
print("xgb_AUC: ", roc_auc_score(y_val, xgb_pred_proba[:,1]))
print("cross_val_best: ", cross_val_score(xgb, x_val, y_val, cv=5).mean())


# %%
### 8.re-fit and submit
rf = RandomForestClassifier(n_estimators=200, max_depth=9, random_state=2022)
rf.fit(X, Y.target)
predicted = rf.predict_proba(X_test)[:,1]

# %%
Y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_test.csv")
roc_auc_score(Y_test.target, predicted )

# %%
