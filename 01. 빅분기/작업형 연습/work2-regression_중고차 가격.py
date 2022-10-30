
# %%
# 1. import library and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBRegressor


# %%
X = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/X_train.csv")
Y = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/y_train.csv")
X_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/X_test.csv")


# %%
# 2. 불필요한 cols 제거
X.drop(columns=['carID'], inplace=True)
X_test.drop(columns=['carID'], inplace=True)

# %%
# 3. na/null값 확인 및 처리
print(X.isna().sum().sum(), X.isnull().sum().sum())
print(X_test.isna().sum().sum(), X_test.isnull().sum().sum())


# %%
# 4. 변수 처리
# display(X.describe(), X.info())
X['year'] = X['year'].astype('object')
X_test['year'] = X_test['year'].astype('object')

col1 = X.select_dtypes(include='object').columns # 명목형 변수
col2 = X.select_dtypes(exclude='object').columns # 수치형 변수

for j in range(len(col1)):
    print(col1[j],"::", X[col1[j]].nunique())
    
for i in range(len(col2)):
    plt.hist(X[col2[i]])
    plt.show()

print("cat",col1)
print("num",col2)

# %%
tmp = X['model'].value_counts()
plt.bar(tmp.index, tmp.values)
plt.show()

stats.chi2_contingency(pd.crosstab(X['brand'], X['model']))[1] # brand와 상관성이 보여 삭제
X.drop(columns=['model'], inplace=True)
X_test.drop(columns=['model'], inplace=True)

col1 = X.select_dtypes(include='object').columns # 명목형 변수
col2 = X.select_dtypes(exclude='object').columns # 수치형 변수

# %%
# 5. dummy and scaling
encoder = LabelEncoder()

X['train'] = 1
X_test['train'] = 0
tmp = pd.concat([X, X_test], axis=0)
for i in range(len(col1)):
    tmp.loc[:,col1[i]] = encoder.fit_transform(tmp.loc[:,col1[i]])

# %%
Dummy = False
if Dummy:
    for idx in range(len(col1)):
        tmp = pd.concat([tmp, pd.get_dummies(tmp[col1[idx]], prefix=col1[idx])], axis=1)
        
    tmp.drop(columns=col1, inplace=True)

minmax_ = False
if minmax_:
    scaler = MinMaxScaler()
    tmp.loc[:, col2] = scaler.fit_transform(tmp.loc[:, col2])
    
X = tmp[tmp['train'] == 1]
X_test = tmp[tmp['train'] == 0]

X.drop(columns=['train'], inplace=True)
X_test.drop(columns=['train'], inplace=True)


# %%
# 6. train test split
x_train, x_val, y_train, y_val = train_test_split(X, Y.price, test_size=0.3)

print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)

# %%
# 7. fit model
rf = RandomForestRegressor(random_state=2022)
rf.fit(x_train, y_train)
cross_val_score(rf, x_val, y_val, scoring='r2').mean()


# %% 
model = XGBRegressor(random_state=2022)
model.fit(x_train, y_train)
cross_val_score(model, x_val, y_val, scoring='r2').mean()


# %%
# gridsearchCV
params = {'n_estimators': [100, 150, 200, 250, 300], 'max_depth':[5,7,9,11], 'max_features':['auto','sqrt']}

grid = GridSearchCV(model, params, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)

# %%
best_params = grid.best_estimator_
predicted = best_params.predict(x_val)

# %%
# 8. re-fit and submit