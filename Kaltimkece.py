# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:02:41 2019

@author: Ferr
"""

import tweepy
import re
import seaborn as sns
import numpy as np
import pandas as pd
import webbrowser
#import json
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

analyser = SentimentIntensityAnalyzer()
translator = Translator()

#Akses untuk Twitter API
ACCESS_TOKEN = '1152912405761888258-gcjN6ML52ustRjM5OwOBW8MupFeqmh'
ACCESS_SECRET = 'pJuurI6rD8EMxWbBQFRIQJ5qFaWHNty5ZbelgpK8S5Clh'
CONSUMER_KEY = 'tFET4xDN214EvZPUNZzTGt8kj'
CONSUMER_SECRET = 'A9DKljNhPiSq0sdk416tiZEJqoDmlpxuGmpdqvCMEcOQsFy2AC'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

#Buat list dari Twitter    
def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw

#Cleaning text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    # remove ascii
    text = _removeNonAscii(text)
    # to lowecase
    text = text.lower()
    return text

def clean_lst(lst):
    lst_new = []
    for r in lst:
        lst_new.append(clean_text(r))
    return lst_new

def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst

def sentiment_analyzer_scores(text,eng=False):
    translator = Translator()
    if eng:
        try:
            text = translator.translate(text).text
        except Exception as e:
            print(str(e))
            
    score = analyser.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

def anl_tweets(lst, title='Tweets Sentiment', engl=True ):
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(
        sents,
        kde=False,
        bins=3)
    ax.set(xlabel='Negative                Neutral                 Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents

user_id = 'kaltimkece'
count = 200

dt_kk = {"raw": pd.Series(list_tweets(user_id, count, True))}
tw_kk = pd.DataFrame(dt_kk)
tw_kk['raw'][3]

tw_kk['clean_text'] = clean_lst(tw_kk['raw'])
tw_kk['clean_text'][3]

tw_kk['clean_vector'] = clean_tweets(tw_kk['clean_text'])
tw_kk['clean_vector'][3]

sentiment_analyzer_scores(tw_kk['clean_text'][3],True)

tw_kk['sentiment'] = pd.Series(anl_tweets(tw_kk['clean_vector'], user_id, True))

#Word Cloud + Sentiment Analysis
stop_words = []

f = open('stopwords-id.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))

#tambahkan kata stopword yang tidak ingin dimunculkan
additional_stop_words = ['t', 'will', 'selengkapnya', 'ketuk']
stop_words += additional_stop_words

def word_cloud(wd_list):
    stopwords = stop_words + list(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=21,
        colormap='jet',
        max_words=50,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear")
    wordcloud.to_file("img/kaltimkece.png");

word_cloud(tw_kk['clean_vector'])
word_cloud(tw_kk['clean_vector'][tw_kk['sentiment'] == 1])
word_cloud(tw_kk['clean_vector'][tw_kk['sentiment'] == -1])

f = open('kaltimkece.html','w')
img = 'img/kaltimkece.png'

message = """<html>
<head></head>
<body>
<p>Tampilkan gambar</p>
<img src="%s"></img>
</body>
</html>"""

whole = message % (img)
f.write(whole)
f.close()

webbrowser.open_new_tab('kaltimkece.html')