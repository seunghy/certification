'''
[참고] - 2회 기출문제 관련
https://www.kaggle.com/datasets/kukuroo3/ecommerce-shipping-data-competition-form
'''
# import library and data
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score

start_time = time.time()

train = pd.read_csv("X_train.csv")
Y = pd.read_csv("y_train.csv")
test = pd.read_csv("X_test.csv")

# print(Y)
# print(train.head())
# print(train.columns)
# print(train.info())
# print(train.describe())

train_ = train.copy()
Y_ = Y.copy()
test_ = test.copy()

# 불필요한 col 제거
test_id = test[['ID']]

train.drop(columns=['ID'], inplace=True)
test.drop(columns=['ID'], inplace=True)

Y_value = Y.drop(columns=['ID'])
Y_value = Y_value.values.ravel()
print(train.select_dtypes(include='object'))


# na/null 처리
# print(train.isna().sum())
# print(test.isna().sum())

# 그외 변수 처리
catcol = train.select_dtypes(include='object').columns

for i in catcol:
    print(train[i].value_counts())
    print(test[i].value_counts())
    
train['Customer_care_calls'] = train['Customer_care_calls'].replace('$7','7').astype('int')
test['Customer_care_calls'] = test['Customer_care_calls'].replace('$7','7').astype('int')

numcol = train.select_dtypes(exclude='object').columns
catcol = train.select_dtypes(include='object').columns
# print(train.describe())

# scaling and labeling
scaler = StandardScaler()
scaler.fit(pd.concat([train[numcol], test[numcol]], axis=0))
train[numcol] = scaler.transform(train[numcol])
test[numcol] = scaler.transform(test[numcol])

for i in catcol:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train[[i]], test[[i]]], axis=0))
    train[i] = encoder.transform(train[[i]])
    test[i] = encoder.transform(test[[i]])

#     train = pd.concat([train, pd.get_dummies(train[i], prefix=i)], axis=1)
#     test = pd.concat([test, pd.get_dummies(test[i], prefix=i)], axis=1)

# train.drop(columns=catcol, inplace=True)
# test.drop(columns=catcol, inplace=True)

print("train.columns:: ",train.columns)

# print(train.describe())


# train test split
x_train, x_val, y_train, y_val = train_test_split(train, Y_value, test_size=0.2, stratify=Y_value, random_state=2022)
print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)

# model fitting
# model1 = LogisticRegression(random_state=2022)
# model1.fit(x_train, y_train)
# print(":::::::logistic::: ", cross_val_score(model1, x_val, y_val, cv=5, scoring='roc_auc').mean())

model2 = RandomForestClassifier(random_state=2022)
model2.fit(x_train, y_train)
print(":::::::rf origin::: ",cross_val_score(model2, x_val, y_val, cv=5, scoring='roc_auc').mean())

# # parameter tuning
# model2_1 = RandomForestClassifier(random_state=2022, max_depth=9)
# model2_1.fit(x_train, y_train)
# print(":::::::rf1::: ",cross_val_score(model2_1, x_val, y_val, cv=5, scoring='roc_auc').mean())

# model2_2 = RandomForestClassifier(random_state=2022, max_depth=11)
# model2_2.fit(x_train, y_train)
# print(":::::::rf2::: ",cross_val_score(model2_2, x_val, y_val, cv=5, scoring='roc_auc').mean())


# model2_3 = RandomForestClassifier(random_state=2022, max_depth=13, n_estimators=300)
# model2_3.fit(x_train, y_train)
# print(":::::::rf3::: ",cross_val_score(model2_3, x_val, y_val, cv=5, scoring='roc_auc').mean())

# pred = model2_3.predict_proba(x_val)[:,1]
# print(roc_auc_score(y_val, pred))


# model2_3 = RandomForestClassifier(random_state=2022, max_depth=15, n_estimators=300)
# model2_3.fit(x_train, y_train)
# print(cross_val_score(model2_3, x_val, y_val, cv=5, scoring='roc_auc').mean())


# re-fit and pred
final_model = RandomForestClassifier(random_state=2022)
final_model.fit(train, Y_value)
pred_prob = final_model.predict_proba(test)[:,1]
pred = final_model.predict(test)

# submit
submit = test_id.copy()
submit.loc[:,'pred'] = pred
submit.columns = Y.columns

print(submit.head())
print(time.time() - start_time)


####################
y_test = pd.read_csv("y_test.csv")
print(roc_auc_score(y_test.iloc[:,1], pred_prob))

