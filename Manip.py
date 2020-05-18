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
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')
def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text


stop_words = stopwords.words('english')


def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]


stemmer = PorterStemmer()


def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))

df = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = df[['headline_text']]
data_text['index'] = data_text.index

nltk.download('punkt')

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
df_merged['tokenized'] = df_merged['headline_text'].apply(apply_all)


print(df_merged.columns)
print(df_merged.columns.values)
print(df_merged)

df_topic3=df_merged.loc[df_merged['Topic'] == 3]
df_topic3=df_topic3.sort_values('publish_date', ascending=True)
df_topic3.to_csv("topic3", index=False)
#df_topic3.to_csv(file_name, sep='\t', encoding='utf-8')
#df_topic3['DATE'] = pd.DatetimeIndex(df['publish_date'], format='%Y%M%D')
print(df_topic3)

#df_merged = pd.merge(df, df_max, on=['document_id','item_id'])
#print (df)
#df_max=df_corp.groupby(['Doc'], sort=False)['Score'].max()
#print(df_max)
