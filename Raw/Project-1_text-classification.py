# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:50:59 2019

@author: Usman
"""

import numpy as np 
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

"""
#importing data set#
reviews = load_files('E:\\Word Embedding\\review_polarity\\txt_sentoken')
x , y = reviews.data, reviews.target

#pickling 

with open('E:\\Word Embedding\\review_polarity\\x.pickle','wb') as f:
    pickle.dump(x,f)
    
with open('E:\\Word Embedding\\review_polarity\\y.pickle','wb') as f:
    pickle.dump(y,f)
"""    
#unpickling
    
with open('E:\\Word Embedding\\review_polarity\\x.pickle','rb') as f:
    x = pickle.load(f)
    
with open('E:\\Word Embedding\\review_polarity\\y.pickle','rb') as f:
    y = pickle.load(f)
    
#cleaning the data
    #Corpus
corpus = []

for i in range(len(x)):
    review = re.sub(r'\W',' ',str(x[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)

#bow model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.7, stop_words = stopwords.words('english'))
x = vectorizer.fit_transform(corpus).toarray()    

#converting bow model into tf-idf model 
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
x = transformer.fit_transform(x).toarray()

"""
for pickling

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.7, stop_words = stopwords.words('english'))
x = vectorizer.fit_transform(corpus).toarray()
"""

#training & testing our model 
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


#Training using logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

#checking if our model is correect
sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

#accurancy of our model.
accurancy = str((cm[0][0]+cm[1][1])/4)+"%"


#saving our model for further use 
#pickleing it

with open("E:\\Word Embedding\\review_polarity\\classifier.pickle","wb") as f:
    pickle.dump(classifier,f)
    
with open("E:\\Word Embedding\\review_polarity\\tfidfmodel.pickle","wb") as f:
    pickle.dump(vectorizer,f)

"""
#loading the model

with open("E:\\Word Embedding\\review_polarity\\classifier.pickle","rb") as f:
    clf = pickle.load(f)
    
with open("E:\\Word Embedding\\review_polarity\\tfidfmodel.pickle","rb") as f:
    tfidf = pickle.load(f)
    
#testing it 
    
s = ["you are a good person","ali is a bad guy"]
s = tfidf.transform(s).toarray()

print(clf.predict(s))    
    
    
    
"""















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
