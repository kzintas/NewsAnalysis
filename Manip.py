import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
#import nltk.stem as stemmer
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk

df = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = df[['headline_text']]
data_text['index'] = data_text.index

df['Doc'] = df.index
#print(df)
#print(df['publish_date'])

#df2= pd.read_csv('corpus_LDA,tsv', error_bad_lines=False);
df_corp = pd.read_csv('corpus.txt', sep="\t")
#print(df_corp)
#df1 = df_corp.loc[df_corp['Doc'] == 2016]
#print(df1)

#Calculates_Max
df_max=df_corp.sort_values('Score', ascending=False).drop_duplicates(['Doc'])
#print(df_max)
df_max=df_max.sort_values('Doc', ascending=True)
#print(df_max)
df_merged=pd.merge(df, df_max, how='left', left_on='Doc', right_on='Doc')
print(df_merged[:10])


pd.set_option('display.max_columns', None)
df_merged.head()
print(df_merged.columns)
print(df_merged.columns.values)
print(df_merged)

df_topic3=df_merged.loc[df_merged['Topic'] == 3]
print(df_topic3)

#df_merged = pd.merge(df, df_max, on=['document_id','item_id'])
#print (df)
#df_max=df_corp.groupby(['Doc'], sort=False)['Score'].max()
#print(df_max)
'''
df['DATE'] = pd.DatetimeIndex(df['publish_date'], format='%Y%M%D')

df['DATE'] = pd.DatetimeIndex(df['publish_date'], format='%Y%M%D')

df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month
df['Day'] = df['DATE'].dt.day

(df
 .assign(MONTH=df['DATE'].dt.strftime('(%m) %B (%y)'))
 .groupby(['NAME', 'MONTH', 'Year'], as_index=False)['SNOW']
 .agg({'AVERAGE': 'mean'})
)
df1 = df.loc[df['Year'] == 2016]
df1.to_csv('average2016.csv', index=False)

df2 = df.loc[df['Year'] == 2017]
df2.to_csv('average2017.csv', index=False)
'''