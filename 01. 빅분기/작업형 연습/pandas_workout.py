'''
pandas 기초 연습
'''
import pandas as pd
import numpy as np

############판다스 연습 튜토리얼 ###########
# Q1
# DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv'
# df = pd.read_csv(DataUrl, sep='\t')
# print(df.head())

# Q2
# print(df.head(5))

# Q3
# print(df.shape)
# print("행: ", df.shape[0])
# print("열: ", df.shape[1])

# Q4
# print(df.columns)

# Q5
# print(df.columns[5])

# Q6
# print(df.iloc[:, 5].dtype)

# Q7
# print(df.index)

# Q8
# print(df.iloc[2,5])

# Q9
# DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv'
# df = pd.read_csv(DataUrl, encoding='cp949')
# print(df.head(2))

# Q10
# print(df.tail(3))

# Q11 ------ check ".select_dtypes(exclude=object)"
# print(df.select_dtypes(exclude=object).columns)

# Q12
# print(df.select_dtypes(include=object).columns)

# Q13
# print(df.isna().sum())

# Q14
# print(df.info())

# Q15
# print(df.describe())

# Q16
# print(df['거주인구'])

# Q17
# print(df['평균 속도'].quantile(0.75) - df['평균 속도'].quantile(0.25))

# Q18
# print(len(df['읍면동명'].unique()))

# Q19
# print(df['읍면동명'].unique())

# Q20
DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv'
df = pd.read_csv(DataUrl)
# print(type(df))

# Q21
# print(df[df['quantity'] == 3][:5])

# Q22
# print(df[df['quantity'] == 3].reset_index(drop=True).head())

# Q23
# print(df[['quantity','item_price']])

# Q24
df['new_price'] = df['item_price'].apply(lambda x: x.split('$')[1]).astype("float")
# print(df['item_price'].head())

# Q25
# Ans = df[df['new_price'] <= 5]
# print(len(Ans))

# Q26
# Ans = df[df['item_name'] == 'Chicken Salad Bowl'].reset_index(drop=True)
# print(Ans)

# Q27
# Ans = df[(df['new_price'] <= 9) & (df['item_name']=='Chicken Salad Bowl')]
# print(Ans)

# Q28
# Ans = df.sort_values('new_price', ascending=True).reset_index(drop=True)
# print(Ans)

# Q29
# Ans =df[df['item_name'].apply(lambda x: 'Chips' in x)] # Ans = df[df['item_name'].str.contains('Chips')]
# print(Ans.head())

# Q30
# print(df[df.columns[::2]])

# Q31
# Ans = df['new_price'].sort_values(ascending=False).reset_index(drop=True)
# print(Ans.head())

# Q32
# Ans = df[df['item_name'].apply(lambda x: x in ['Steak Salad', 'Bowl'])]
# print(Ans)

# Q33
# tmp = Ans = df[df['item_name'].apply(lambda x: x in ['Steak Salad', 'Bowl'])]
# print(tmp.drop_duplicates('item_name'))

# Q34
# tmp = Ans = df[df['item_name'].apply(lambda x: x in ['Steak Salad', 'Bowl'])]
# print(tmp.drop_duplicates('item_name', keep='last'))

# Q35
# Ans = df[df['new_price'] >= df['new_price'].mean()]
# print(Ans)

# Q36
# df.loc[df.item_name == 'Izze', 'item_name'] = 'Fizzy Lizzy'
# print(df.head())

# Q37
# print(df.choice_description.isnull().sum())

###########
# Q38
# df.loc[df.choice_description.isnull(), 'choice_description'] = 'NoData'
# print(df.head())

# Q39
# print(df[df['choice_description'].str.contains("Black")])

# Q40
# Ans = df[~df['choice_description'].str.contains("Vegetables")]
# print(len(Ans))

# Q41
# print(df[df['item_name'].apply(lambda x: x[0] == 'N')])

# Q42
# print(df[df['item_name'].apply(lambda x: len(x)>=15)])

# Q43
# Ans = df[df['new_price'].isin([1.69, 2.39, 3.39, 4.45, 9.25, 10.98, 11.75, 16.98])]
# print(len(Ans))

########
# Q44
DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/AB_NYC_2019.csv'
df = pd.read_csv(DataUrl)
# print(df.head(5))

# Q45 --------- ".sort_index()"
# print(df['host_name'].value_counts().sort_index())

# Q46 -------------
# Ans = df['host_name'].value_counts().sort_values(ascending=False)
# Ans = pd.DataFrame(Ans)
# Ans.columns = ['counts']
# Ans.index.name = "host_name"
# print(Ans.head(5))

# Q47
# Ans = df.groupby(['neighbourhood_group','neighbourhood']).size().reset_index(drop=False)
# Ans.rename(columns={Ans.columns[-1]:'size'}, inplace=True)
# print(Ans)

# Q48 ----------- "as_index=False"
# Ans = df.groupby(['neighbourhood_group','neighbourhood'], as_index=False).size()
# Ans = Ans.groupby(['neighbourhood_group'], as_index=False).max()
# print(Ans)

# Q49
# Ans = df.groupby(['neighbourhood_group'])['price'].agg(['mean','var','max','min'])
# print(Ans)

# Q50
# Ans = df[['neighbourhood_group','reviews_per_month']].groupby('neighbourhood_group').agg(['mean','var','max','min'])
# print(Ans)

# Q51
# Ans = df.groupby(['neighbourhood','neighbourhood_group'])['price'].mean()
# print(Ans)

# Q52 --------------- "unstack"
# Ans = df.groupby(['neighbourhood','neighbourhood_group'])['price'].mean()
# print(Ans.unstack())

# Q53
# Ans = df.groupby(['neighbourhood','neighbourhood_group'])['price'].mean().unstack()
# Ans.fillna(-999.0, inplace=True)
# print(Ans)

# Q54

# Q55
# Q56
# Q57