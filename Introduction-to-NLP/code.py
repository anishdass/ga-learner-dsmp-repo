# --------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix


# Code starts here

news = pd.read_csv(path)
news = news[['TITLE','CATEGORY']]
dist = news.CATEGORY.value_counts()
print(dist)
print(news.head())

# load data


# subset data

# distribution of classes


# display class distribution


# display data


# Code ends here


# --------------
# Code starts here

# stopwords
stop = set(stopwords.words('english'))

# retain only alphabets
news.TITLE = news.TITLE.apply(lambda x: re.sub("[^a-zA-Z]", " ", x))

# convert to lowercase and tokenize
news.TITLE = news.TITLE.apply(lambda x: x.lower().split())

# remove stopwords
news.TITLE = news.TITLE.apply(lambda x: [i for i in x if i not in stop])

# join list elements
news.TITLE = news.TITLE.apply(lambda x: ' '.join(x))

# split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(news.TITLE, news.CATEGORY, test_size = 0.2, random_state = 3)

# Code ends here


# --------------
# Code starts here

# initialize count vectorizer
count_vectorizer = CountVectorizer()

# initialize tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# fit and transform with count vectorizer
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# fit and transform with tfidf vectorizer
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Code ends here


# --------------
# Code starts here

# initialize multinomial naive bayes
nb_1 = MultinomialNB()
nb_2 = MultinomialNB()

# fit on count vectorizer training data
nb_1.fit(X_train_count, Y_train)

# fit on tfidf vectorizer training data
nb_2.fit(X_train_tfidf, Y_train)

# accuracy with count vectorizer
acc_count_nb = accuracy_score(nb_1.predict(X_test_count), Y_test)

# accuracy with tfidf vectorizer
acc_tfidf_nb = accuracy_score(nb_2.predict(X_test_tfidf), Y_test)

# display accuracies
print(acc_count_nb, acc_tfidf_nb)

# Code ends here


# --------------
import warnings
warnings.filterwarnings('ignore')

# initialize logistic regression
logreg_1 = OneVsRestClassifier(LogisticRegression(random_state=10))
logreg_2 = OneVsRestClassifier(LogisticRegression(random_state=10))

# fit on count vectorizer training data
logreg_1.fit(X_train_count, Y_train)

# fit on tfidf vectorizer training data
logreg_2.fit(X_train_tfidf, Y_train)


# accuracy with count vectorizer
acc_count_logreg = accuracy_score(logreg_1.predict(X_test_count), Y_test)

# accuracy with tfidf vectorizer
acc_tfidf_logreg = accuracy_score(logreg_2.predict(X_test_tfidf), Y_test)


# display accuracies
print(acc_count_logreg, acc_tfidf_logreg)

# Code ends here


