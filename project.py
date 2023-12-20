# Author - Shamura Ahmad

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

#train and test text
train_texts = data.text[:5000]
test_texts = data.text[5000:]

#define count-vectorizer 
cv = CountVectorizer(min_df=0, max_df=1, binary = False, ngram_range = (1,3))

#apply fit-tranform to train text
cv_train_texts = cv.fit_transform(train_texts)

#apply transform to test texts
cv_test_texts = cv.transform(test_texts)

#define label-binarizer 
lb = LabelBinarizer()

#apply fit-transform to train word label
lb_train_label = lb.fit_transform(train_word_label)

#apply fit-transform to test word label
lb_test_label = lb.fit_transform(test_word_label)

#define multinomial naive bayes,predic data and check accuracy
mnb = MultinomialNB()
mnb_bow = mnb.fit(cv_train_texts, lb_train_label)
mnb_predict = mnb.predict(cv_test_texts)
mnb_score = accuracy_score(lb_test_label, mnb_predict)
print("Accuracy (MNB) :", mnb_score)

#define support vector classifier,predic data and check accuracy
svm = SVC()
svm_bow = svm.fit(cv_train_texts, lb_train_label)
svm_predict = svm.predict(cv_test_texts)
svm_score = accuracy_score(lb_test_label, svm_predict)
print("Accuracy (SVM) :", svm_score)

#apply wordcloud to get a visual representation of the most used words from real news
real_news = " ".join(list(data[data['word_label'] == 'real']['text']))
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(real_news)
mp.figure(figsize=(10, 7))
mp.imshow(wordcloud, interpolation="bilinear")
mp.axis('off')
mp.show()
