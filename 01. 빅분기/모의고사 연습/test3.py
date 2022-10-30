'''
[실기모의고사]-3회차
https://www.datamanim.com/dataset/practice/q3.html
'''

# %%
# 작업형 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Question1
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/spotify/spotify.csv')

df.dropna(inplace=True)
# df.sort_values('top year', ascending=True, inplace=True)
# df.reset_index(drop=True, inplace=True)
# df['rank'] = 1
# rank = 1
# idx = 0
# while True:
#     if df.loc[idx, 'year released'] == df.loc[idx+1, 'year released']:
#         rank += 1 
#     else:
#         rank = 1
        
#     df.loc[idx+1, 'rank'] = rank 
    
#     idx += 1  
#     if idx == len(df)-1:
        
#         break

# Ans = df[(~df['year released'].isna())&(df['rank'] == 1)]
# Ans = Ans['bpm'].mean()
# print(Ans)

df['rank'] = list(range(1, 101))*10
Ans = df[df['rank'] == 1]['bpm'].mean()
print(Ans)


# %%
# Question2
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/spotify/spotify.csv')

df = df[df['top year'] == 2015]
Ans = df['artist'].value_counts(ascending=False).index[0]
print(Ans)

# %%
# Question3
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/spotify/spotify.csv')

