
# %%
## 1. import library and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%%
X = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_train.csv")
Y = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_test.csv")

# %%
## 2. na/null값 확인
print(X.isna().sum().sum(), X.isnull().sum().sum())
print(Y.isna().sum().sum(), Y.isnull().sum().sum())
print(test.isna().sum().sum(), test.isnull().sum().sum())

# %%
## 3. 불필요한 cols 삭제
X = X.drop(columns=['StudentID'])
X_test = test.drop(columns=['StudentID'])

# %%
## 4. 변수값 확인 및 전처리
# X.info()

label_col = X.select_dtypes(include='object').columns
for i in range(len(label_col)):
    X[label_col[i]] = X[label_col[i]].str.replace(" ","")
    print(X[label_col[i]].value_counts())
    print(test[label_col[i]].value_counts())

# %%
## 5. dummy and scaling
scaler = MinMaxScaler()
X[X.select_dtypes(exclude='object').columns] = scaler.fit_transform(X[X.select_dtypes(exclude='object').columns])
X_test[X_test.select_dtypes(exclude='object').columns] = scaler.transform(X_test[X_test.select_dtypes(exclude='object').columns])

# %%
# X = pd.get_dummies(X)
# X_test = pd.get_dummies(X_test)
encoder = LabelEncoder()
label_cols = X.select_dtypes(include='object').columns

for j in range(len(label_cols)):
    X.loc[:,label_cols[j]] = encoder.fit_transform(X.loc[:,label_cols[j]])
    X_test.loc[:,label_cols[j]] = encoder.transform(X_test.loc[:,label_cols[j]])


# %%
## 6. train test split
x_train, x_val, y_train, y_val = train_test_split(X, Y.G3, test_size=0.3)

print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)


# %%
## 7. model fit
rf = RandomForestRegressor(random_state=2022)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_val)

print(rf.score(x_val, y_val))
print(r2_score(y_val, rf_pred))
print(cross_val_score(rf, x_val, y_val, scoring='r2').mean())

print("XGB")
xgb = XGBRegressor(random_state=2022)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_val)

print(xgb.score(x_val, y_val))
print(r2_score(y_val, xgb_pred))
print(cross_val_score(xgb, x_val, y_val, scoring='r2').mean())


# %%
## 8. re-fit and submit
model = XGBRegressor(random_state=2022)
model.fit(X, Y.G3)
model_pred = model.predict(X_test)

predicted = pd.DataFrame({'StudentID': test.StudentID, "predict":model_pred})
real = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_test.csv")

tmp = pd.merge(predicted, real, on='StudentID', how='outer')
r2_score(real['G3'], model_pred)

# %%

