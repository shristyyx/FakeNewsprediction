#this model will predict if a news is fake or real
#this model works on Logistic Regression model as the value is either fake or not (0 or 1)

#importing the dependencies
import numpy as np
import pandas as pd
import re #regular expressions : used for searching words
from nltk.corpus import stopwords #used to remove unwanted words like is, the, etcimp
from nltk.stem.porter import PorterStemmer #removes suffix and prefix from a word
from sklearn.feature_extraction.text import TfidfVectorizer #converts text data to feature vectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

#loading dataset in pandas
fake_news_dataset = pd.read_csv('/content/train.csv')
fake_news_dataset.shape

fake_news_dataset.head()

#count missing values in dataset
fake_news_dataset.isnull().sum()

#replacing the missing adatset with emoty strings
fake_news_dataset= fake_news_dataset.fillna('')
fake_news_dataset.isnull().sum()

fake_news_dataset['content'] = fake_news_dataset['author'] +" "+ fake_news_dataset['title']
print(fake_news_dataset['content'])

# Stemming : reducing a word to it's root word
port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ',
                             content)  # only alphabets remain, replaces comma, numbers, etc with ' ',
    stemmed_content = stemmed_content.lower()  #
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

fake_news_dataset['content'] = fake_news_dataset['content'].apply(stemming)
print(fake_news_dataset['content'])

#seperating the data nad label columns
X = fake_news_dataset['content'].values
Y = fake_news_dataset['label'].values

#converting text to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X= vectorizer.transform(X)
print(X)

#splitting the data to train and test data

X_train, X_test , Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=5, stratify = Y)
#stratify maintains a proportion

#training the model

model = LogisticRegression()
model.fit(X_train, Y_train)

#Evaluation

#accuracy score on training data
X_train_pred = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_pred, Y_train)
print(train_data_accuracy)

#accuracy score on testing data
X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)
print(test_data_accuracy)

#predictive model
n = 100
X_new = X_test[n]
predict_X = model.predict(X_new)
print(predict_X)
print(Y_test[n])