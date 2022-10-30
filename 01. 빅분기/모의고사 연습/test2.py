'''
[실기모의고사]-2회차
https://www.datamanim.com/dataset/practice/q2.html
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb

# %%
# 작업 1유형

# Question1
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv")
def digit_check(value):
    tmp = ''
    for i in value:
        if i.isdigit():
            tmp += i
    return tmp
    
df['age'] = df['age'].map(digit_check).astype('float')
Ans = df[df['gender'] == 'Male']['age'].mean()
print(Ans)

# %%
# Question2
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv")

med_ = df['bmi'].median()
df.loc[df['bmi'].isna(), 'bmi'] = med_
Ans = round(df['bmi'].mean(), 3)
print(Ans)

# %%
# Question3
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv")
df['bmi'].fillna(method='ffill', inplace=True)
Ans = round(df['bmi'].mean(), 3)
print(Ans)

# %%
# Question4
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv")

def digit_check(value):
    tmp = ''
    for i in value:
        if i.isdigit():
            tmp += i
    return tmp

df['age'] = df['age'].map(digit_check).astype('int')
df['age2'] = (df['age'] / 10).astype('int')
df['age2'] = (df['age2']*10)
dic = df.groupby('age2')['bmi'].mean().to_dict()

#################
# def fill(df):
#     dff = df[df['bmi'].isna()].copy()
        
#     for i in range(len(dff)):
#         idx = dff.index[i]
#         dff.loc[idx, 'bmi'] = dic[dff.loc[idx, 'age2']]
#     return dff

# df = fill(df)
#################

#################
idx = df[df['bmi'].isna()].copy()
# df.loc[df['bmi'].isna(),'bmi'] = 
df.loc[df['bmi'].isna(), 'bmi'] = df[df['bmi'].isna()]['age2'].map(lambda x: dic[x])
#################

Ans = round(df['bmi'].mean(), 3)
print(Ans)

# %%
# Question5
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv")

df.loc[df['avg_glucose_level'] >= 200,'avg_glucose_level'] = 199
Ans = df[df['stroke'] == 1]['avg_glucose_level'].mean()
Ans = round(Ans, 3)
print(Ans)


# %%
# Question6
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv")

df2 = df.sort_values('Attack', ascending=False).reset_index()
Ans = df2.loc[:400,'Legendary'].sum() - df2.loc[400:,'Legendary'].sum()
print(Ans)

# %%
# Question7
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv")

Ans = df.groupby('Type 1')['Total'].mean().sort_values(ascending=False).index[2]
print(Ans)

# %%
# Question8
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv")

df.dropna(inplace=True)
df = df.reset_index(drop=True)
idx = int(len(df)*0.6)
df2 = df.iloc[:idx,:].copy()
Ans = np.quantile(df2['Defense'], 0.25)
print(Ans)

# %%
# Question9
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv")

mean_ = df[df['Type 1'] == 'Fire']['Attack'].mean()
Ans = df[(df['Type 1'] == 'Water')&(df['Attack']>=mean_)]
print(len(Ans))



# %%
# Question10
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv")

df['abs'] = abs(df['Speed'] - df['Defense'])
Ans = df.loc[df['abs'].argmax(),'Generation']
print(Ans)

# %%
# 작업 2유형
train= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv')
test= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/test.csv')

# 불필요한 col 제거
testId = test.id
Y = train.stroke
train.drop(columns=['id','stroke'], inplace=True)
test.drop(columns=['id'], inplace=True)


# 변수 처리
# age: to int
# print(train['age'].unique(), test['age'].unique())
def to_digit(value):
    tmp = ''
    for i in value:
        if i.isdigit():
            tmp += i
    return int(tmp)

train['age'] = train['age'].map(to_digit)
train['age'] = ((train['age'] // 10) * 10).astype('string')

# na 값 처리
print(train.isna().sum().sum()) # bmi 컬럼 결측치 :52
print(test.isna().sum().sum()) # bmi 컬럼 결측치 :57

if 'bmi' in train.columns:
    med_ = pd.concat([train['bmi'], test['bmi']], axis=0).median()
    train.loc[train['bmi'].isna(), 'bmi'] = med_
    test.loc[test['bmi'].isna(), 'bmi'] = med_
    
numcol = train.select_dtypes(exclude='object').columns
catcol = train.select_dtypes(include='object').columns

    
# scaling and categorical variable
scaler = StandardScaler()
for i in numcol:
    scaler.fit(pd.concat([train[i], test[i]], axis=0).values.reshape(-1,1))
    train[i] = scaler.transform(train[i].values.reshape(-1,1))
    test[i] = scaler.transform(test[i].values.reshape(-1, 1))
    

for i in catcol:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train[i], test[i]], axis=0))
    train[i] = encoder.transform(train[i])
    test[i] = encoder.transform(test[i])
    print("unique:", train[i].unique())


# for i in catcol:
#     onehot = OneHotEncoder()
#     tmp = onehot.fit(pd.concat([train[i], test[i]], axis=0).values.reshape(-1, 1)).toarray()
#     tmp = pd.DataFrame(tmp, columns = [i+str(num) for num in range(tmp.shape[1])])

#     tmp = onehot.fit_transform(test[i].values.reshape(-1,1)).toarray()
#     tmp = pd.DataFrame(tmp, columns = [i+str(num) for num in range(tmp.shape[1])])
#     train = pd.concat([test, tmp], axis=1)

# train.drop(columns=catcol, inplace=True)
# test.drop(columns=catcol, inplace=True)
    
    
# pd.DataFrame(tmp.toarray(), columns=[catcol[1]+str(i) for i in range(tmp.toarray().shape[1])])

# train test split
x_train, x_val, y_train, y_val = train_test_split(train, Y, test_size=0.3, stratify=Y)

# model fit
model1 = XGBClassifier(random_state=2022)
model1.fit(x_train, y_train)
print("model1: ",cross_val_score(model1, x_val, y_val, cv=3, scoring='roc_auc').mean())

model2 = lgb.LGBMClassifier(random_state=2022)
print("model2: ",cross_val_score(model2, x_val, y_val, cv=3, scoring='roc_auc').mean())

model3 = RandomForestClassifier(random_state=2022)
print("model3: ",cross_val_score(model3, x_val, y_val, cv=3, scoring='roc_auc').mean())

# re-fit and submit
# %%
