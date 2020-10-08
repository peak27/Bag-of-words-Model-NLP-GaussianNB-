# Natural Language Processing (NLP)
# Step 1

#   Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing Datset (tsv file)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #    Ignoring " with quoting = 3

#   Cleaning the text
import re 
import nltk
#   nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    feedback = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #   Keeping texts only
    feedback = feedback.lower() #   All Lowercase
    feedback = feedback.split() #   Creating list of review words
    ps = PorterStemmer()
    # Remove stopwords & Stemming (Keeping root word only )
    feedback = [ps.stem(word) for word in feedback if not word in set(stopwords.words('english'))]
    feedback = ' '.join(feedback)
    corpus.append(feedback)

#   Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
# Keeping only frequent 1500 words from corpus
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#   Spliting training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#   Fitting Naive Bayes to Our Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#   Predicting results for test set
y_hat = classifier.predict(X_test)

#   Confusion Matrix
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_hat)


















