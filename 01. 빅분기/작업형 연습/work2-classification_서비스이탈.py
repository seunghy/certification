
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score, roc_auc_score, f1_score

# %%
# ### 1. 서비스 이탈예측 데이터
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv")

# %%
# na/null값 체크
print(x_train.isna().sum().sum())
print(x_train.isnull().sum().sum())

print(y_train.isna().sum().sum())
print(y_train.isnull().sum().sum())

print(x_test.isna().sum().sum())
print(x_test.isnull().sum().sum())

# %%
# 불필요한 cols 제거
x_train.drop(columns=['Surname'], inplace=True)
x_test.drop(columns=['Surname'], inplace=True)

# %%
# category/수치형 변수들 전처리
x_train['Gender'] = x_train['Gender'].str.replace(" ",'')
x_test['Gender'] = x_test['Gender'].str.replace(" ",'')

x_train['Gender'] = x_train.apply(lambda x: 'Male' if x['Gender'] in ['male','Male'] else 'Female', axis=1)
x_test['Gender'] = x_test.apply(lambda x: 'Male' if x['Gender'] in ['male','Male'] else 'Female', axis=1)


# %%
# category변수들을 dummy화
label_col = ['Geography','Gender','IsActiveMember']

x_train['train'] = 1
x_test['train'] = 0

X = pd.concat([x_train, x_test], axis=0)

for i in range(len(label_col)):
    dummy = pd.get_dummies(X[label_col[i]], prefix=label_col[i])
    X = pd.concat([X, dummy], axis=1)

X.drop(columns=label_col, inplace=True)

x_train = X[X['train'] == 1]
x_test = X[X['train'] == 0]

x_train.drop(columns=['train'], inplace=True)
x_test.drop(columns=['train'], inplace=True)

# %%
x_train2, x_val, y_train2, y_val  = train_test_split(x_train.iloc[:,1:], y_train.iloc[:,1], test_size=0.3, stratify=y_train.iloc[:,1])

# ## model1
# rf = RandomForestClassifier(random_state=2022)
# rf.fit(x_train2, y_train2)
# rf_pred = rf.predict(x_val)
# rf_pred_prob = rf.predict_proba(x_val)[:,1]

# print("-------RF")
# print("roc: ", roc_auc_score(y_val, rf_pred_prob))
# print("cross_val: ", cross_val_score(rf, x_val, y_val, scoring="roc_auc").mean())
# print("acc:", rf.score(x_val, y_val))
# print("cross_val: ", cross_val_score(rf, x_val, y_val).mean())
# print("f1 score: ", f1_score(y_val, rf_pred))
# print(confusion_matrix(y_val, rf_pred))
# # pd.DataFrame({'col':x_train2.columns, 'imp':rf.feature_importances_}).sort_values("imp", ascending=False)

# ## model2
# XGB = XGBClassifier(random_state=2022)
# XGB.fit(x_train2, y_train2)
# XGB_pred = XGB.predict(x_val)
# XGB_pred_prob = XGB.predict_proba(x_val)[:,1]

# print("-----XGB")
# print("roc: ", roc_auc_score(y_val, XGB_pred_prob))
# print("cross_val: ", cross_val_score(XGB, x_val, y_val, scoring="roc_auc").mean())
# print("acc:", XGB.score(x_val, y_val))
# print("cross_val: ", cross_val_score(XGB, x_val, y_val).mean())
# print("f1 score: ", f1_score(y_val, XGB_pred))
# print(confusion_matrix(y_val, XGB_pred))

# ## model3
# lm = LogisticRegression(random_state=2022)
# lm.fit(x_train2, y_train2)
# lm_pred = lm.predict(x_val)
# lm_pred_prob = lm.predict_proba(x_val)[:,1]

# print("-----LM")
# print("roc: ", roc_auc_score(y_val, lm_pred_prob))
# print("cross_val: ", cross_val_score(lm, x_val, y_val, scoring="roc_auc").mean())
# print("acc:", lm.score(x_val, y_val))
# print("cross_val: ", cross_val_score(lm, x_val, y_val).mean())
# print("f1 score: ", f1_score(y_val, lm_pred))

# %%
# 결과물 제출용
rf = RandomForestClassifier(random_state=2022)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)



# %%
y_test = pd.read_csv('/kaggle/input/churn-model-data-set-competition-form/test_label/y_test.csv')
print("final out: ", roc_auc_score(y_test, rf_pred))