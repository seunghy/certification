'''
[실기모의고사]-1회차
https://www.datamanim.com/dataset/practice/q1.html
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from scipy.stats import shapiro, chi2_contingency

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

################################ 작업형 1 ################################
# %%
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

# Question1
def age_func(age):
    if 0 <= age and age<10:
        return 0
    elif 10<= age and age < 20:
        return 10
    elif 20<= age and age < 30:
        return 20
    elif 30<= age and age < 40:
        return 30
    elif 40<= age and age < 50:
        return 40
    elif 50<= age and age < 60:
        return 50
    elif 60<= age and age < 70:
        return 60
    elif 70<= age and age < 80:
        return 70
    elif 80<= age and age < 90:
        return 80
    else:
        return 90
    
df['age2'] = df['age'].map(age_func)
Ans = df['age2'].value_counts().index[0]
print(Ans)

# %%
# Question2
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

def age_func(age):
    if 0 <= age and age<10:
        return 0
    elif 10<= age and age < 20:
        return 10
    elif 20<= age and age < 30:
        return 20
    elif 30<= age and age < 40:
        return 30
    elif 40<= age and age < 50:
        return 40
    elif 50<= age and age < 60:
        return 50
    elif 60<= age and age < 70:
        return 60
    elif 70<= age and age < 80:
        return 70
    elif 80<= age and age < 90:
        return 80
    else:
        return 90
    
df['age2'] = df['age'].map(age_func)
Ans = df['age2'].value_counts().max()
print(Ans)

# %%
# Question3
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

Ans = df[(25<=df['age']) & (df['age']<29)&(df['housing'] == 'yes')]
print(len(Ans))

# %%
# Question4
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

df2 = df.select_dtypes(include='object')
colname = ''
num = 0
for i in range(len(df2.columns)):
    if df2[df2.columns[i]].nunique() > num:
        colname = df2.columns[i]
        num = df2[df2.columns[i]].nunique()
print(colname)

# %%
# Question5
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

Ans = df[df['balance'] >= df['balance'].mean()].sort_values("ID", ascending=False)
Ans = Ans['balance'][:100].mean()
print(Ans)

# %%
# Question6
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

Ans = df.groupby(['day','month']).size().sort_values(ascending=False).index[0]
print(Ans)

# %%
# Question7
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

df2 = df[df['job'] == 'unknown']['age']
Ans = shapiro(df2)[1]
print(Ans)

# %%
# Question8
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

Ans = df[['age','balance']].corr().iloc[0,1]
print(Ans)

# %%
# Question9
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

t = pd.crosstab(df.y, df.education)
Ans = chi2_contingency(t)[1]
print(Ans)

# %%
# Question10 -- 다시
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

df2 = df.groupby(['job','marital']).size().reset_index()
df2.rename(columns={0:'cnt'}, inplace=True)
df2 = df2.pivot_table(index='job', columns='marital')
df2.fillna(0, inplace=True)
df2.columns = [c[1] for c in df2.columns.values]
df2['ratio'] = df2['divorced'] / df2['married']
Ans = df2.sort_values('ratio', ascending=False).iloc[0,-1]
print(Ans)

################################ 작업형 2 ################################
# 모델링 및 submission파일 생성까지
# 성능지표: roc_auc
# %%
start_time = time.time()

train= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
test= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/test.csv')
submission= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/submission.csv')

# 불필요한 변수 삭제
testId = test.ID
Y = train.y

train.drop(columns=['ID','y'], inplace=True)
test.drop(columns="ID", inplace=True)

# null/na값 확인
print("train null:",train.isna().sum().sum())
print("test null: ",test.isna().sum().sum())

# dummy and scaling
catcol = train.select_dtypes(include='object').columns
numcol = train.select_dtypes(exclude='object').columns

encoder = LabelEncoder()
Y = encoder.fit_transform(Y.values)

for i in catcol:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train[i], test[i]], axis=0))
    train[i] = encoder.transform(train[i].values)
    test[i] = encoder.transform(test[i].values)
    
for j in numcol:
    scaler = StandardScaler()
    scaler.fit(pd.concat([train[[j]], test[[j]]], axis=0))
    
    train[j] = scaler.transform(train[[j]])
    test[j] = scaler.transform(test[[j]])
    
# 기타 변수 처리

# train test split
x_train, x_val, y_train, y_val = train_test_split(train, Y, test_size=0.3, stratify=Y, random_state=2022)

# # modeling
model1 = RandomForestClassifier()
model1.fit(x_train, y_train)
model1_pred = model1.predict_proba(x_val)
# print("model1 roc: ",roc_auc_score(y_val, model1_pred[:,1]))
print(cross_val_score(model1, x_val, y_val, scoring='roc_auc').mean())

model2 = XGBClassifier()
model2.fit(x_train, y_train)
model2_pred = model2.predict_proba(x_val)
# print("model2 roc: ", roc_auc_score(y_val, model2_pred[:,1]))
print(cross_val_score(model2, x_val, y_val, scoring='roc_auc').mean())

# # Gridsearch 
# params = {'max_depth':[3,5,7,9], 'n_estimators':[100, 150, 200, 250, 300], 'max_features':['auto','sqrt']}
# grid = GridSearchCV(model1, params, cv=5)
# grid.fit(x_train, y_train)
# print("best_params_: ",grid.best_params_)

# best_param = grid.best_estimator_
# best_param_pred = best_param.predict_proba(x_val)[:,1]
# print("best_param_pred: ",best_param_pred)
# print(cross_val_score(best_param, x_val, y_val, scoring='roc_auc').mean())

# # refit and submit
# params = {'max_depth':[3,5,7,9], 'n_estimators':[100, 150, 200, 250, 300], 'max_features':['auto','sqrt']}
# grid = GridSearchCV(model1, params, cv=5)
# grid.fit(train, Y)

# submit
model = RandomForestClassifier()
model.fit(train, Y)
pred_prob = model.predict_proba(test)

submit = pd.DataFrame({'ID':testId.values,'predict':pred_prob[:,1]})
submit.to_csv("submit_test1.csv", index=False)

print("Running time: ", time.time() - start_time)
# %%
