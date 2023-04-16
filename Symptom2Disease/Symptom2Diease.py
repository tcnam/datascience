# %%
import pandas as pd
import numpy as np
import pathlib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df=pd.read_csv(f'Symptom2Disease.csv',sep=',', encoding='utf-8', index_col=0)
df.head()
# %%
def splitWords(df):
    newText=[]
    for value in df['text']:
        temp=[re.sub('[^a-zA-Z]',' ',value).lower().split()]
        newText.append(temp)
        del(temp)
    # resultDf['newText']=[re.sub('[^a-zA-Z]',' ',value) for value in resultDf['text']]
    # resultDf['newText']=[value.lower() for value in resultDf['newText']]
    # resultDf['newText']=[value.split() for value in resultDf['newText']]
    feature=pd.DataFrame(newText,columns=['newText'])
    df = df.reset_index(drop=True)
    resultDf=pd.concat([df,feature], axis=1)
    print(resultDf.head())
    return resultDf
# print(result['newText'])

# %%
def removeStopWords(df):
    wordnet=WordNetLemmatizer()
    newText2=[]
    for value in df['newText']:
        temp=[[wordnet.lemmatize(word) for word in value if word not in set(stopwords.words('english'))]]
        newText2.append(temp)
        del(temp)
    feature=pd.DataFrame(newText2,columns=['newText2'])
    df = df.reset_index(drop=True)
    resultDf=pd.concat([df,feature], axis=1)
    print(resultDf.head())
    return resultDf
# %%
result=splitWords(df)
result2=removeStopWords(result)