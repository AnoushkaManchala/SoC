#Q1
import pandas as pd
#Q2
print(pd.__version__)
#Q3
pd.show_versions()
#Q4
import numpy as np

data = {
    'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
    'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
    'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(data, index=labels)
#Q5
df.info()
#Q6
df.iloc[:3]
#Q7
df[['animal', 'age']]
#Q8
df.loc[df.index[[3, 4, 8]], ['animal', 'age']]
#Q9
df[df['visits'] > 3]
# This will return an empty DataFrame since max visits is 3. To show rows with 3+:
# df[df['visits'] >= 3]
#Q10
df[df['age'].isna()]
#Q11
df[(df['animal'] == 'cat') & (df['age'] < 3)]
#Q12
df[df['age'].between(2, 4)]
#Q13
df.at['f', 'age'] = 1.5
#Q14
df['visits'].sum()
#Q15
df.groupby('animal')['age'].mean()
#Q16
df.loc['k'] = ['dog', 5.0, 2, 'yes']
df = df.drop('k')
#Q17
df['animal'].value_counts()
#Q18
df.sort_values(by=['age', 'visits'], ascending=[False, True])
#Q19
df['priority'] = df['priority'].map({'yes': True, 'no': False})
#Q20
df['animal'] = df['animal'].replace('snake', 'python')
#Q21
df.pivot_table(values='age', index='animal', columns='visits', aggfunc='mean')
#Q22
df[df['A'].shift() != df['A']]
#Q23
df.sub(df.mean(axis=1), axis=0)
#Q24
df.sum().idxmin()
#Q25
df.drop_duplicates().shape[0]
#Q26
df.apply(lambda row: row[row.isna()].index[2], axis=1)
#Q27
df.groupby('grps')['vals'].apply(lambda x: x.nlargest(3).sum())
#Q28
bins = pd.cut(df['A'], bins=range(0, 101, 10))
df.groupby(bins)['B'].sum()




