# author - Shamura Ahmad

#importing libraries 
import pandas as pd
import numpy as nm
import nltk
import matplotlib.pyplot as mp
import nltk.tokenize as nt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

#setting data and url 
url = "https://raw.githubusercontent.com/raima2001/HerWILL-NSDC-DS-Contest/main/news_dataset_subset%20(1).csv"
data = pd.read_csv(url)

#observing first 5 rows of the data
data.head()

#summary of the data
data.describe()

#counting unique values 
unique_value_count = data['word_label'].value_counts()
unique_value_count

#plotting word label distribution
Count = unique_value_count
WordLabel = data['word_label'].unique()
mp.title('Distribution of the Word Label')
mp.bar(WordLabel, Count, width=0.5)
mp.xlabel('Word Label')
mp.ylabel('Count')
mp.show()

#getting information about the data
data_details = data.info()
data_details

#counting null values
data['text'].isnull().sum()

#filling nill values with 'No info'
data['text'] = data['text'].fillna('No info')
data['text'].isnull().sum()

#changing datatype
data['text'] = data['text'].astype(str)

#applying word tokenize
nltk.download('punkt')
data['text'] = data['text'].apply(word_tokenize)

#removing punctuation
print(" ".join(data['text'][1]))
data['text'] = data['text'].apply(lambda x: [item for item in x if item.isalpha()])
print(" ".join(data['text'][1]))

#converting to lowercase 
print(" ".join(data['text'][1]))
data['text'] = data['text'].apply(lambda x: [item.lower() for item in x])
print(" ".join(data['text'][1]))

#removing filter words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(" ".join(data['text'][2]))
data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])
print(" ".join(data['text'][2]))

#applying lemmitizer 
nltk.download('wordnet')
lem = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x: [lem.lemmatize(item) for item in x])

#applying porter-stemmer 
ps = PorterStemmer()
data['text'] = data['text'].apply(lambda x: [ps.stem(item) for item in x])
data['text'] = data['text'].apply(lambda x: " ".join(x))

train_texts = data.text[:5000]
