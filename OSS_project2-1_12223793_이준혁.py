#task 1

import pandas as pd

df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
stats = ['H', 'avg', 'HR', 'OBP']

for y in df['year'].unique():
    if 2015 <= y <= 2018:
        for i in stats:
            print("%d" %y)
            tp = df[df['year'] == y].nlargest(10, i)[['batter_name', i]]
            print(tp)
            print("\n")

#task 2

df = df[df['year'] == 2018]
ps = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

for p in ps:
    tp = df[df['cp'] == p].nlargest(1, 'war')[['batter_name', 'war']]
    print("%s"%p)
    print(tp)
    print("\n")


#task 3
    
df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

stats = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']

cor = df[stats + ['salary']].corr()['salary'].loc[stats]
print(cor)


ms = cor.idxmax()
print("%s가 연봉과 가장 관련있는 지표입니다." %ms)


