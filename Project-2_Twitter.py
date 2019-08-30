# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:13:17 2019

@author: Usman
"""

import tweepy
import re 
import pickle

from tweepy import OAuthHandler

# initializing the keys 

consumer_key=''
consumer_secret=''
access_token=''
access_secret = ''

#setting up the OAuth Handler 

#this authenticates the twitter app
auth = OAuthHandler(consumer_key,consumer_secret)
#this authenticates the rights to fetch the tweets
auth.set_access_token(access_token, access_secret)

#this is the search keyword
args = ['facebook']

api = tweepy.API(auth,timeout=10)

list_tweets = []

query = args[0]


if len(args) ==1:
    for status in tweepy.Cursor(api.search, q=query+ " -filter.retweets",lang = 'en',result_type = 'recent'.items(100)):
        list_tweets.append(status.text)


with open('tfidfmodel.pickle','rb') as f:
    vect = pickle.load(f)
    
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    
    
#for plotting afterwards
sentpos = 0
sentneg = 0    
    

#filtering/ Preprocessing the tweets    
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","are not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    sentiment = clf.predict(vect.transform(tweet).toarray())
    #print(tweet,":",sentiment)
    #adding total pos , neg tweets
    if sentiment[0] == 1:
        sentpos = sentpos + 1
    else:
        sentneg = sentneg + 1
        
#plotting
        
import matplotlib.pyplot as plt
import numpy as np

objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[sentpos,sentneg],alpha = 0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title("Number of Positive & Negative Tweets")
plt.show()