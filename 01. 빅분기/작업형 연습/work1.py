
# %%
from calendar import c
import pandas as pd
import numpy as np
from torch import QInt32Storage

############작업 1유형 #########
### 1. 유튜브 인기동영상 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv",index_col=0)

# # 인기동영상 제작횟수가 많은 채널 상위 10개명을 출력하라 (날짜기준, 중복포함)
# idx = df['channelId'].value_counts().sort_values(ascending=False)[:10].index
# Ans = df.loc[df['channelId'].isin(idx),'channelTitle'].unique()
# print(Ans)

# # 논란으로 인기동영상이 된 케이스를 확인하고 싶다. dislikes수가 like 수보다 높은 동영상을 제작한 채널을 모두 출력하라
# Ans = df[df.apply(lambda x: x['dislikes'] > x['likes'], axis=1)]['channelTitle'].unique()
# print(Ans)

# # 채널명을 바꾼 케이스가 있는지 확인하고 싶다. channelId의 경우 고유값이므로 이를 통해 채널명을 한번이라도 바꾼 채널의 갯수를 구하여라
# tmp = df[['channelTitle','channelId']].drop_duplicates()
# Ans = (tmp.groupby('channelId')['channelTitle'].size() > 1).sum()
# print(Ans)

# # 일요일에 인기있었던 영상들중 가장많은 영상 종류(categoryId)는 무엇인가?
# tmp = df.copy()
# tmp['trending_date2'] = pd.to_datetime(tmp['trending_date2'])
# tmp = tmp[tmp['trending_date2'].dt.weekday == 6]
# Ans = tmp['categoryId'].value_counts().sort_values(ascending=False)
# print(Ans.index[0])

# 각 요일별 인기 영상들의 categoryId는 각각 몇개 씩인지 하나의 데이터 프레임으로 표현하라 ----------**
# tmp = df.copy()
# tmp['trending_date2'] = pd.to_datetime(tmp['trending_date2'])
# tmp['trending_date2'] = tmp['trending_date2'].dt.day_name()
# Ans = tmp.groupby(['trending_date2','categoryId'], as_index=False).size()
# Ans = Ans.pivot(index='categoryId', columns='trending_date2')
# print(Ans)

# %%
# # 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다. viewcount대비 댓글수가 가장 높은 영상을 확인하라 (view_count값이 0인 경우는 제외한다)
# tmp = df.copy()
# tmp = tmp[tmp['view_count'] != 0].reset_index(drop=True)
# tmp['ratio'] = tmp['comment_count'] / tmp['view_count']
# tmp.sort_values('ratio', ascending=False).iloc[0,:]


# %%
# # like 대비 dislike의 수가 가장 적은 영상은 무엇인가? (like, dislike 값이 0인경우는 제외한다)
# tmp = df.copy()
# tmp = tmp[(tmp['likes'] != 0) & (tmp['dislikes'] != 0)].reset_index(drop=True)
# tmp['dislikes_ratio'] = tmp['dislikes'] / tmp['likes']
# tmp.sort_values('dislikes_ratio', ascending=True).iloc[0,:]


# %%
# # 가장많은 트렌드 영상을 제작한 채널의 이름은 무엇인가? (날짜기준, 중복포함)
# tmp = df.copy()
# idx = tmp.groupby('channelId').size().sort_values(ascending=False).index[0]
# print(tmp[tmp.channelId==idx]['channelTitle'].unique()[0])

# %%
# # 20회(20일)이상 인기동영상 리스트에 포함된 동영상의 숫자는?
# tmp = df.copy()
# Ans = tmp[['title', 'channelId']].value_counts().sort_values(ascending=False) >= 20
# print(Ans.sum())

# %%
### 2. 유튜브 공범컨텐츠 동영상 데이터
# channel =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/channelInfo.csv')
# video =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/videoInfo.csv')
# display(channel.head())
# display(video.head())

# %%
# # 각 데이터의 ‘ct’컬럼을 시간으로 인식할수 있게 datatype을 변경하고 video 데이터의 videoname의 각 value 마다 몇개의 데이터씩 가지고 있는지 확인하라
# channel['ct'] = pd.to_datetime(channel['ct'])
# video['ct'] = pd.to_datetime(video['ct'])

# Ans = video['videoname'].value_counts()
# print(Ans)

# %%
# # 수집된 각 video의 가장 최신화 된 날짜의 viewcount값을 출력하라
# Ans = video.sort_values(['videoname','ct'], ascending=True).reset_index(drop=True)
# Ans.drop_duplicates('videoname', keep='last')[['viewcnt','videoname','ct']].reset_index(drop=True)

# %%
# # Channel 데이터중 2021-10-03일 이후 각 채널의 처음 기록 됐던 구독자 수(subcnt)를 출력하라
# tmp = channel[channel['ct'] > pd.to_datetime('2021-10-03')]
# Ans = tmp.sort_values(['ct','channelid'])
# Ans.drop_duplicates('channelid', keep='first')[['channelname', 'subcnt']].reset_index(drop=True)

# %%
# # 각채널의 2021-10-03 03:00:00 ~ 2021-11-01 15:00:00 까지 구독자수 (subcnt) 의 증가량을 구하여라
# tmp = channel[(channel['ct'] >= pd.to_datetime('2021-10-03 03:00:00'))&(channel['ct'] <= pd.to_datetime('2021-11-01 15:00:00'))]
# Ans = tmp.sort_values(['channelid','ct'])
# start = Ans.drop_duplicates('channelid', keep='first')[['channelname','subcnt']].reset_index(drop=True)
# start.columns = ['channelname','start']

# end = Ans.drop_duplicates('channelid', keep='last')[['channelname','subcnt']].reset_index(drop=True)
# end.columns = ['channelname','end']

# Ans = pd.merge(start, end, how='left')
# Ans['del'] = Ans['end'] - Ans['start']
# Ans = Ans[['channelname','del']]
# Ans

# %% -- PASSS
# # 각 비디오는 10분 간격으로 구독자수, 좋아요, 싫어요수, 댓글수가 수집된것으로 알려졌다. 공범 EP1의 비디오정보 데이터중 수집간격이 5분 이하, 20분이상인 데이터 구간( 해당 시점 전,후) 의 시각을 모두 출력하라
# tmp = video[video['videoname'] == '공범 EP1'].reset_index(drop=True)


# %%
# # 각 에피소드의 시작날짜(년-월-일)를 에피소드 이름과 묶어 데이터 프레임으로 만들고 출력하라
# video['date'] = video['ct'].dt.date
# Ans = video.sort_values(['videoname','date'], ascending=True)[['date','videoname']]
# Ans.drop_duplicates('videoname', keep='first')

# %%
# # video 정보의 가장 최근 데이터들에서 각 에피소드의 싫어요/좋아요 비율을 ratio 컬럼으로 만들고 videoname, ratio로 구성된 데이터 프레임을 ratio를 오름차순으로 정렬하라
# video['date'] = video['ct'].dt.date
# Ans = video.sort_values(['videoname','date'], ascending=True)
# Ans = Ans.drop_duplicates('videoname', keep='last')
# Ans['ratio'] = Ans['dislikecnt'] / Ans['likecnt']
# Ans[['videoname','ratio']].sort_values('ratio', ascending=True).reset_index(drop=True)

# %%
# # 2021-11-01 00:00:00 ~ 15:00:00까지 각 에피소드별 viewcnt의 증가량을 데이터 프레임으로 만드시오
# tmp = video[(video['ct'] >= pd.to_datetime('2021-11-01 00:00:00'))&(video['ct'] <= pd.to_datetime('2021-11-01 15:00:00'))]
# Ans = tmp.sort_values(['videoname', 'ct'])

# start = Ans.drop_duplicates('videoname', keep='first')[['videoname','viewcnt']]
# start.columns = ['videoname','start']

# end = Ans.drop_duplicates('videoname', keep='last')[['videoname','viewcnt']]
# end.columns = ['videoname','end']

# Ans = pd.merge(start, end)
# Ans['viewcnt'] = Ans['end'] - Ans['start']
# Ans = Ans[['videoname','viewcnt']]
# Ans

# %%
# # video 데이터 중에서 중복되는 데이터가 존재한다. 중복되는 각 데이터의 시간대와 videoname 을 구하여라
# Ans = video[video[['videopk','ct','videoname']].duplicated()][['videoname','ct']]
# print(Ans)

# %%
### 3. 월드컵 출전선수 골기록 데이터
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')

# %%
# # 주어진 전체 기간의 각 나라별 골득점수 상위 5개 국가와 그 득점수를 데이터프레임형태로 출력하라
# tmp = df.copy()
# Ans = tmp.groupby('Country')['Goals'].sum().sort_values(ascending=False)
# print(pd.DataFrame(Ans).head())

# %%
# # 주어진 전체기간동안 골득점을 한 선수가 가장 많은 나라 상위 5개 국가와 그 선수 숫자를 데이터 프레임 형식으로 출력하라
# tmp = df.copy()
# country = tmp.groupby('Country')['Goals'].sum().sort_values(ascending=False).head().index

# Ans = tmp[tmp['Country'].apply(lambda x: x in country)]
# Ans.groupby('Country')['Player'].size().sort_values(ascending=False)

# %%
# # Years 컬럼은 년도 -년도 형식으로 구성되어있고, 각 년도는 4자리 숫자이다. 년도 표기가 4자리 숫자로 안된 케이스가 존재한다. 해당 건은 몇건인지 출력하라
# tmp = df.copy()
# tmp['new'] = tmp['Years'].apply(lambda x: x.split('-'))

# cnt = 0
# for i in range(len(tmp)):
#     for j in range(len(tmp.loc[i, 'new'])):
#         if len(str(tmp.loc[i, 'new'][j])) != 4:
#             cnt += 1
# print(cnt)

# %%
# # **Q3에서 발생한 예외 케이스를 제외한 데이터프레임을 df2라고 정의하고 데이터의 행의 숫자를 출력하라 (아래 문제부터는 df2로 풀이하겠습니다) **
# tmp = df.copy()
# tmp['new'] = tmp['Years'].apply(lambda x: x.split('-'))

# for i in range(len(tmp)):
#     for j in range(len(tmp.loc[i, 'new'])):
#         if len(str(tmp.loc[i, 'new'][j])) != 4:
#             tmp.loc[i,'check'] = True
#             break
#         else:
#             tmp.loc[i,'check'] = False

# df2 = tmp[tmp['check'] == False].reset_index(drop=True).drop(columns=['check'])
# df2.head()

# %%
# # 월드컵 출전횟수를 나타내는 ‘LenCup’ 컬럼을 추가하고 4회 출전한 선수의 숫자를 구하여라
# tmp = df2.copy()
# tmp['LenCup'] = tmp['new'].str.len()
# (tmp.groupby('Player')['LenCup'].sum() == 4).sum() # tmp['LenCup'].value_counts()

# %%
# # Yugoslavia 국가의 월드컵 출전횟수가 2회인 선수들의 숫자를 구하여라
# tmp = df2.copy()
# tmp['LenCup'] = tmp['new'].str.len()
# Ans = tmp[(tmp['Country'] == 'Yugoslavia') & (tmp['LenCup'] == 2)]
# print(len(Ans))

# %%
# # 2002년도에 출전한 전체 선수는 몇명인가?
# tmp = df2.copy()
# Ans = tmp[tmp['Years'].str.contains('2002')]
# print(len(Ans))

# %%
# # 이름에 ‘carlos’ 단어가 들어가는 선수의 숫자는 몇 명인가? (대, 소문자 구분 x)
# tmp = df2.copy()
# (tmp['Player'].str.lower().str.contains('carlos')).sum()


# %%
# # 월드컵 출전 횟수가 1회뿐인 선수들 중에서 가장 많은 득점을 올렸던 선수는 누구인가?
# tmp = df2.copy()
# tmp = tmp[tmp['new'].str.len() == 1]
# tmp.sort_values('Goals', ascending=False)['Player'].values[0]

# %%
# # 월드컵 출전횟수가 1회 뿐인 선수들이 가장 많은 국가는 어디인가?
# tmp = df2.copy()
# tmp[tmp['new'].str.len() == 1]['Country'].value_counts().index[0]

# %%
### 4. 서울시 따릉이 이용정보 데이터
# df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')

# %%
# # 대여일자별 데이터의 수를 데이터프레임으로 출력하고, 가장 많은 데이터가 있는 날짜를 출력하라
# result = df.groupby('대여일자').size().to_frame()
# result.sort_index(inplace=True)
# result.columns = ['대여일자']
# result

# Ans = result.index[result['대여일자'].argmax()]
# print(Ans)

# %%
# # 각 일자의 요일을 표기하고 (‘Monday’ ~’Sunday’) ‘day_name’컬럼을 추가하고 이를 이용하여 각 요일별 이용 횟수의 총합을 데이터 프레임으로 출력하라
# df['day_name'] = pd.to_datetime(df['대여일자']).dt.day_name()
# df['day_name'].value_counts().to_frame()

# %%
# # 각 요일별 가장 많이 이용한 대여소의 이용횟수와 대여소 번호를 데이터 프레임으로 출력하라
# Ans = df.groupby(['day_name','대여소번호'], as_index=False).size().sort_values(['day_name','size'], ascending=False)
# Ans.drop_duplicates('day_name', keep='first').reset_index(drop=True)

# %%
# # 나이대별 대여구분 코드의 (일일권/전체횟수) 비율을 구한 후 가장 높은 비율을 가지는 나이대를 확인하라. 일일권의 경우 일일권 과 일일권(비회원)을 모두 포함하라
# tmp = df[df['대여구분코드'].isin(['일일권','일일권(비회원)'])].reset_index(drop=True)

# cnt = tmp.groupby(['연령대코드']).size()
# total = df.groupby(['연령대코드']).size()

# cnt/total


# %%
# # 연령대별 평균 이동거리를 구하여라
# df.groupby('연령대코드')['이동거리'].agg('mean')

# %%
# # 연령대 코드가 20대인 데이터를 추출하고,이동거리값이 추출한 데이터의 이동거리값의 평균 이상인 데이터를 추출한다.최종 추출된 데이터를 대여일자, 대여소 번호 순서로 내림차순 정렬 후 1행부터 200행까지의 탄소량의 평균을 소숫점 3째 자리까지 구하여라
# tmp = df[df['연령대코드'] == '20대']
# result = tmp[tmp['이동거리']>=tmp['이동거리'].mean()]

# Ans = result.sort_values(['대여일자','대여소번호'], ascending=False)['탄소량'][:200].values
# Ans = list(map(float, Ans))
# print(round(sum(Ans)/len(Ans), 3))

# %%
# # 6월 7일 ~10대의 “이용건수”의 중앙값은?
# tmp = df.copy()
# tmp['대여일자'] = pd.to_datetime(tmp['대여일자'])
# tmp = df[(df['연령대코드'] == '~10대')&(tmp['대여일자'] == pd.to_datetime('2021-06-07'))]
# tmp['이용건수'].median()

# %% ------------------------------------------- groupby.head()
# # 평일 (월~금) 출근 시간대(오전 6,7,8시)의 대여소별 이용 횟수를 구해서 데이터 프레임 형태로 표현한 후 각 대여시간별 이용 횟수의 상위 3개 대여소와 이용횟수를 출력하라
# tmp = df[df['대여시간'].isin([6,7,8])].reset_index(drop=True)
# Ans = tmp.groupby(['대여시간','대여소번호']).size().to_frame('이용 횟수')
# Ans = Ans.sort_values(['대여시간','이용 횟수'], ascending=False)
# Ans.groupby('대여시간').head(3)

# %%
# # 이동거리의 평균 이상의 이동거리 값을 가지는 데이터를 추출하여 추출데이터의 이동거리의 표본표준편차 값을 구하여라
# tmp = df[df['이동거리'] >= df['이동거리'].mean()].reset_index(drop=True)
# tmp['이동거리'].std()

# %%
# # 남성(‘M’ or ‘m’)과 여성(‘F’ or ‘f’)의 이동거리값의 평균값을 구하여라
# tmp = df.copy()
# tmp['sex'] = tmp['성별'].map(lambda x: '남' if x in ['M','m'] else "여")
# tmp.groupby('sex')['이동거리'].mean().to_frame()

# %%
# ### 5. 전세계 행복도 지표 데이터
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv',encoding='utf-8')

# %%
# # 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 각년도의 행복랭킹 10위를 차지한 나라의 행복점수의 평균을 구하여라
# tmp1 = df[df['년도'] == 2018]
# result1 = tmp1[tmp1['행복랭킹'] == 10]['점수'].values[0]

# tmp2 = df[df['년도'] == 2019]
# result2 = tmp2[tmp2['행복랭킹'] == 10]['점수'].values[0]

# print((result1+result2)/2)


# %%
# # 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 각년도의 행복랭킹 50위이내의 나라들의 각각의 행복점수 평균을 데이터프레임으로 표시하라
# tmp = df[df['행복랭킹']<=50].reset_index(drop=True)
# tmp.groupby('년도')['점수'].agg('mean').to_frame()

# %%
# # 2018년도 데이터들만 추출하여 행복점수와 부패에 대한 인식에 대한 상관계수를 구하여라
# tmp1 = df[df['년도'] == 2018]
# tmp1[['점수','부패에 대한인식']].corr().iloc[0,1]

# %%
# # 2018년도와 2019년도의 행복랭킹이 변화하지 않은 나라명의 수를 구하여라
# tmp1 = df[df['년도'] == 2018]['나라명'].values.tolist()
# tmp2 = df[df['년도'] == 2019]['나라명'].values.tolist()

# cnt = 0
# for i in range(len(tmp1)):
#     if tmp1[i] == tmp2[i]:
#         cnt += 1
# print(cnt)

# %%
# # 2019년도 데이터들만 추출하여 각변수간 상관계수를 구하고 내림차순으로 정렬한 후 상위 5개를 데이터 프레임으로 출력하라. 컬럼명은 v1,v2,corr으로 표시하라
# tmp = df[df['년도'] == 2019]
# result = tmp.corr().stack().reset_index()
# result.columns = ['v1','v2','corr']
# result = result[(result['corr'] < 1.0) & (result['corr'] > -1.0)]
# result.sort_values('corr', ascending=False).reset_index(drop=True)

# %%
# # 각 년도별 하위 행복점수의 하위 5개 국가의 평균 행복점수를 구하여라
# tmp = df.groupby('년도').tail()
# tmp.groupby('년도')['점수'].agg('mean')

# %%
# # 2019년 데이터를 추출하고 해당데이터의 상대 GDP 평균 이상의 나라들과 평균 이하의 나라들의 행복점수 평균을 각각 구하고 그 차이값을 출력하라
# tmp = df[df['년도'] == 2019]
# value1 = tmp[tmp['상대GDP'] >= tmp['상대GDP'].mean()]['점수'].mean()
# value2 = tmp[tmp['상대GDP'] <= tmp['상대GDP'].mean()]['점수'].mean()
# value1 - value2

# %%
# # 각년도의 부패에 대한인식을 내림차순 정렬했을때 상위 20개 국가의 부패에 대한인식의 평균을 구하여라
# result = df.sort_values('부패에 대한인식',ascending=False).groupby('년도').head(20)
# result.groupby('년도')['부패에 대한인식'].mean()

# %%
# # 2018년도 행복랭킹 50위 이내에 포함됐다가 2019년 50위 밖으로 밀려난 국가의 숫자를 구하여라
# value1 = set(df[(df['년도']==2018) & (df['행복랭킹'] <= 50)]['나라명'])
# value2 = set(df[(df['년도']==2019) & (df['행복랭킹'] <= 50)]['나라명'])

# len(value1 - value2)

# %% --------- 다시
# 2018년,2019년 모두 기록이 있는 나라들 중 년도별 행복점수가 가장 증가한 나라와 그 증가 수치는?
tmp = df.copy()
country = set(tmp[tmp['년도']==2018]['나라명']) & set(tmp[tmp['년도']==2019]['나라명'])
country = list(country)

df2 = df[df['나라명'].isin(country)].reset_index(drop=True)
df2[df2['년도'] == 2018]['점수'] = df2[df2['년도'] == 2018]['점수'].values * (-1)
df2.groupby(['나라명']).agg('sum')['점수'].sort_values().to_frame().iloc[-1]

# %%
# ### 6. 지역구 에너지 소비량 데이터





# %%
# 포켓몬 정보 데이터
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv')

# Q1
# Ans = df[df['Legendary'] == True]['HP'].mean() - df[df['Legendary'] == False]['HP'].mean()
# print(abs(Ans))

# Q2
# Ans = df['Type 2'].value_counts().sort_values(ascending=False).index[0]
# print(Ans)

# Q3
# jong = df['Type 1'].value_counts().sort_values(ascending=False).index[0]
# Ans = df[df['Type 1'] == jong]['Attack'].mean() / df[df['Type 1'] == jong]['Defense'].mean()
# print(Ans)

# Q4
# Ans = df[df['Legendary'] == True]['Generation'].value_counts().sort_values(ascending=False).index[0]
# print(Ans)

# Q5
# corrdf = df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].corr()
# corrdf = corrdf.unstack().reset_index()
# corrdf.columns = ['level0','level1','corr']
# corrdf = corrdf[corrdf.apply(lambda x:x['corr'] != 1, axis=1)]
# corrdf.sort_values('corr', ascending=False, inplace=True)

# Ans = corrdf.iloc[corrdf['corr'].argmax(),:]
# print(Ans)


# Q6
# newdf = df.sort_values(['Generation','Attack'], ascending=False).groupby(['Generation']).head(5)
# Ans = newdf['Attack'].mean()
# print(Ans)

# Q7
# tmp = df.groupby(['Type 1' , 'Type 2']).size().sort_values(ascending=False)
# print(tmp.head(1))

# Q8
# tmp = df.groupby(['Type 1' , 'Type 2']).size().sort_values(ascending=False).to_frame()
# ans = tmp[tmp.loc[:,0] == 1].loc[:,0].sum()
# print(ans)

# %%
# Q9
tmp = df.groupby(['Type 1' , 'Type 2']).size().sort_values(ascending=False).to_frame()
tmp[tmp.loc[:,0] == 1].loc[:,0]

# tmp = df.groupby(['Generation','Type 1' , 'Type 2']).size().reset_index()
# tmp['Generation'].value_counts()




# %%
# 대한민국 체력장 데이터
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/body/body.csv')

# Q1
Ans = np.mean(df['수축기혈압(최고) : mmHg'] - df['이완기혈압(최저) : mmHg'])
print(Ans)

# Q2
Ans = df[(df['측정나이'] >= 50)&(df['측정나이']<=59)]['신장 : cm'].mean()
print(Ans)

# Q3
df['연령대'] = (df['측정나이'] // 10) * 10
Ans = df['연령대'].value_counts()

# Q4
df['연령대'] = (df['측정나이'] // 10) * 10
Ans = df.groupby(['연령대','등급'],as_index=False).size()
print(Ans)

# Q5
male = df[df['측정회원성별'] == 'M']
Ans = male[male['등급'] == 'A']['체지방율 : %'].mean() - male[male['등급'] == 'D']['체지방율 : %'].mean()
print(abs(Ans))

# Q6
female = df[df['측정회원성별'] == 'F']
Ans = female[female['등급'] == 'A']['체중 : kg'].mean() - female[female['등급'] == 'D']['체중 : kg'].mean()
print(abs(Ans))

# Q7
df['bmi'] = df['체중 : kg'] / ((df['신장 : cm']/100) ** 2)
Ans = df[df['측정회원성별'] == 'M']['bmi'].mean()
print(Ans)

# Q8
df['bmi'] = df['체중 : kg'] / ((df['신장 : cm']/100) ** 2)
Ans = df[df['체지방율 : %'] > df['bmi']]['체중 : kg'].mean()
print(Ans)

# Q9
Ans = df[df['측정회원성별'] == 'M']['악력D : kg'].mean() - df[df['측정회원성별'] == 'F']['악력D : kg'].mean()
print(abs(Ans))

# Q10
Ans = df[df['측정회원성별'] == 'M']['교차윗몸일으키기 : 회'].mean() - df[df['측정회원성별'] == 'F']['교차윗몸일으키기 : 회'].mean()
print(abs(Ans))

# %%
# 기온 강수량 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/weather/weather2.csv")

# Q1
df['time'] = pd.to_datetime(df['time'])
df['time2'] = df['time'].dt.month
newdf = df[df['time2'].isin([6,7,8])]
Ans = newdf[newdf['이화동기온'] > newdf['수영동기온']]
print(len(Ans))

# Q2
Ans1 = df.loc[df['이화동강수'].argmax(), 'time']
Ans2 = df.loc[df['수영동강수'].argmax(), 'time']
print(Ans1, Ans2)



# %%
# 성인 건강검진 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv")

# Q1
df['혈압차'] = df['수축기혈압']  - df['이완기혈압']
Ans = df.groupby('연령대코드(5세단위)')['혈압차'].agg('var').sort_values(ascending=False).index[4].head(1)
print(Ans)

# Q2
df['WHtR'] = df['허리둘레'] / df['신장(5Cm단위)']
newdf = df[df['WHtR']>=0.58]
tmp = newdf.groupby('성별코드').size()
Ans = tmp['M'] / tmp['F']
print(Ans)



# %%
# 서비스 이탈예측 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv")

# Q1
male = df[df['Gender'] == 'Male']
Ans = male.groupby('Geography')['Exited'].sum().sort_values(ascending=False).head(1)

# Q2
Ans = df[(df['HasCrCard'] == 1)&(df['IsActiveMember'] == 1)]['Age'].mean()
print(round(Ans, 4))

# Q3
Ans = df[df['Balance']>=df['Balance'].median()]['CreditScore'].std()
print(round(Ans, 3))


# %%
# 자동차 보험가입 예측데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv")

# Q1
med_ = df['Annual_Premium'].median()
newdf = df[df['Vehicle_Age'] == '> 2 Years']
Ans = newdf[newdf['Annual_Premium'] >= med_]['Vintage'].mean()
print(Ans)

# Q2
Ans = df.pivot_table(index='Vehicle_Age', columns='Gender', values='Annual_Premium',aggfunc='mean')
print(Ans)

# %%
# 핸드폰 가격 예측데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv")

# Q1
df.groupby(['price_range','n_cores']).size().sort_values(ascending=False).groupby('price_range').head(1)

# Q2
newdf = df[df['price_range'] == 3]
corr = newdf.corr().unstack().to_frame()
corr = corr[corr.loc[:,0] != 1].sort_values(0, ascending=False)
corr.loc[corr.index[0],:]


# %%
# 비행탑승 경험 만족도 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv")

newdf = df[df['Arrival Delay in Minutes'].isna()]
# newdf.groupby(['Class','satisfaction']).size().sort_values(ascending=False)
newdf = pd.crosstab(newdf['Class'], newdf['satisfaction'])
tmp = newdf.apply(lambda x: x['neutral or dissatisfied']<x['satisfied'], axis=1).sort_values(ascending=False)

Ans = tmp.index[tmp.argmax()]
print(Ans)


# %%
# 수질 음용성 여부 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/train.csv")

newdf = df[['ph']].dropna()
value = np.quantile(newdf, 0.25)
Ans = newdf[newdf['ph']<=value].mean()
print(Ans[0])

# %%
# 의료 비용 예측 데이터
# train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/train.csv")
# up = len(train) * 0.7

# no = train[train['smoker'] == 'no']
# yes = train[train['smoker'] == 'yes']

# value1 = np.quantile(no['charges'], 0.9)
# value2 = np.quantile(yes['charges'], 0.9)

# no = no[no['charges']>=value1]
# yes = yes[yes['charges']>=value2]

# print(abs(no['charges'].mean() - yes['charges'].mean()))


# %%
# 킹카운티 주거지 가격예측문제 데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/kingcountyprice//train.csv")

df = df[df['bedrooms'] == df['bedrooms'].mode()[0]]
Ans = np.quantile(df.price, 0.1) - np.quantile(df.price, 0.9)
print(abs(Ans))

# %%
# 대학원 입학가능성 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/train.csv")

# Y = df[['Chance of Admit']]
# df.drop(columns=['Serial No.','Chance of Admit'], inplace=True)

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()
# rf.fit(df, Y)
# Ans = pd.DataFrame({'importance':rf.feature_importances_}, index=df.columns)
# print(Ans.sort_values('importance',ascending=False))

# %%
# 레드 와인 퀄리티 예측 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/train.csv")

# df3 = df[df['quality'] == 3]
# df8 = df[df['quality'] == 8]

# diff = abs(df3.describe().loc['std',:] - df8.describe().loc['std',:])
# Ans = diff.index[diff.argmax()]
# print(Ans)

# %%
# 약물 분류 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/train.csv")

# df['Age2'] = (df['Age'] // 10) * 10
# df = df[df['Sex'] == 'M']
# Ans = df.groupby('Age2')['Na_to_K'].mean().to_frame()
# print(Ans)

# %%
# 사기회사 분류 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/train.csv")

# Ans = df.groupby('Risk')['Score_A','Score_B'].mean()
# print(Ans)


# %%
# 현대 차량 가격 분류문제 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/train.csv")

# lst = list(df['model'].value_counts(ascending=False)[:3].index)
# print(lst)

# df = df[df['model'].apply(lambda x: x in lst)]
# Ans = df.groupby('model')['price'].mean().to_frame()
# print(Ans)


# %%
# 당뇨여부판단 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/train.csv")

# tmp = df.groupby('Outcome').agg('mean')
# Ans = tmp.apply(lambda x: x[1]-x[0])
# print(Ans)

# %%
# ### 넷플릭스 주식 데이터
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/nflx/NFLX.csv") 

# # 매년 5월달의 open가격의 평균값을 데이터 프레임으로 표현하라
# df['Date'] = pd.to_datetime(df['Date'])
# df['Date'] = df['Date'].dt.strftime('%Y-%m')
# Ans = df.groupby('Date')['Open'].mean().to_frame().reset_index()
# Ans = Ans[Ans.apply(lambda x: x['Date'].split('-')[1] == '05', axis=1)]
# Ans.set_index('Date', inplace=True)
# print(Ans)

# %%
