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
          title="Barchart Tweets of @"+title)
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
plt.savefig("img/kaltimkece_bar.png");


labels = tw_kk['sentiment'].astype('category').cat.categories.tolist()
counts = tw_kk['sentiment'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=["Negative", "Netral", "Positive"], colors=['red', 'blue', 'green'], autopct='% 1.1f %%') #autopct is show the % on plot
ax1.axis('equal')
plt.title('Piechart Tweets @kaltimkece')
plt.savefig("img/kaltimkece_pie.png");


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
    
def word_cloud_p(wd_list):
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
    wordcloud.to_file("img/kaltimkece_p.png");

def word_cloud_n(wd_list):
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
    wordcloud.to_file("img/kaltimkece_n.png");

from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import nltk
import string

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# email module has some useful functions
import os, sys, email,re



from sklearn.feature_extraction.text import TfidfVectorizer
data = tw_kk['clean_vector'] #df['body_new']


tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',#tokenizer = tokenize_and_stem,
                             max_features = 20000)
tf_idf = tf_idf_vectorizor.fit_transform(data)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()




class Kmeans:
    """ K Means Clustering
    
    Parameters
    -----------
        k: int , number of clusters
        
        seed: int, will be randomly set if None
        
        max_iter: int, number of iterations to run algorithm, default: 200
        
    Attributes
    -----------
       centroids: array, k, number_features
       
       cluster_labels: label for each data point
       
    """
    
    def __init__(self, k, seed = None, max_iter = 200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter
        
            
    
    def initialise_centroids(self, data):
        """Randomly Initialise Centroids
        
        Parameters
        ----------
        data: array or matrix, number_rows, number_features
        
        Returns
        --------
        centroids: array of k centroids chosen as random data points 
        """
        
        initial_centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_centroids]

        return self.centroids
    
    
    def assign_clusters(self, data):
        """Compute distance of data from clusters and assign data point
           to closest cluster.
        
        Parameters
        ----------
        data: array or matrix, number_rows, number_features
        
        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster
            
        """
        
        if data.ndim == 1:
            data = tw_kk['sentiment']
        
        dist_to_centroid =  pairwise_distances(data, self.centroids, metric = 'euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)
        
        return  self.cluster_labels
    
    
    def update_centroids(self, data):
        """Computes average of all data points in cluster and
           assigns new centroids as average of data points
        
        Parameters
        -----------
        data: array or matrix, number_rows, number_features
        
        Returns
        -----------
        centroids: array, k, number_features
        """
        
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis = 0) for i in range(self.k)])
        
        return self.centroids
    
    
    
    def predict(self, data):
        """Predict which cluster data point belongs to
        
        Parameters
        ----------
        data: array or matrix, number_rows, number_features
        
        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster
        """
        
        return self.assign_clusters(data)
    
    def fit_kmeans(self, data):
        """
        This function contains the main loop to fit the algorithm
        Implements initialise centroids and update_centroids
        according to max_iter
        -----------------------
        
        Returns
        -------
        instance of kmeans class
            
        """
        self.centroids = self.initialise_centroids(data)
        
        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)          
            if iter % 100 == 0:
                print("Running Model Iteration %d " %iter)
        print("Model finished running")
        return self   
    
    

sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

number_clusters = range(1, 7)

kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters]
kmeans

score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]
print(score)

plt.plot(number_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Method')
plt.savefig('img/kmeanst.png', dpi=100)
plt.show()

sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)
test_e = Kmeans(3, 1, 600)
fitted = test_e.fit_kmeans(Y_sklearn)
predicted_values = test_e.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis')

centers = fitted.centroids
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
plt.savefig('img/sebaran.png', dpi=100)
plt.show()

#fig1.savefig('tessstttyyy.png', dpi=100)

word_cloud(tw_kk['clean_vector'])
word_cloud_p(tw_kk['clean_vector'][tw_kk['sentiment'] == 1])
word_cloud_n(tw_kk['clean_vector'][tw_kk['sentiment'] == -1])

f = open('kaltimkece.html','w')
img5 = 'img/kaltimkece_bar.png'
img4 = 'img/kaltimkece_pie.png'
img1 = 'img/kaltimkece.png'
img2 = 'img/kaltimkece_p.png'
img3 = 'img/kaltimkece_n.png'
img6 = 'img/kmeanst.png'
img7 = 'img/sebaran.png'

message = """<html>
<head><title>WordCloud Sentimen KaltimKece</title></head>
<body>
<h2>Sebaran kata dalam Twitter @kaltimkece</h2>
<div class="row">
<div class="column"><img src="%s"></img></div>
<br><br>
<div class="column"><img src="%s"></img></div>
</div>
<h2 align="center">Frekuensi kata dalam Twitter @kaltimkece</h2>
<img src="%s" width="1280" heigth="1024" align="center"></img>
<br><br>
<h2 align="center">Frekuensi kata sentimen Positif terhadap Twitter @kaltimkece</h2>
<img src="%s" width="1280" heigth="1024" align="center"></img>
<br><br>
<h2 align="center">Frekuensi kata sentimen Negatif terhadap Twitter @kaltimkece</h2>
<img src="%s" width="1280" heigth="1024" align="center"></img>
<br><br>
<h2 align="center">KMeans Elbow method terhadap Twitter @kaltimkece</h2>
<img src="%s" ></img>
<br><br>
<h2 align="center">Sebaran titik sentimen Twitter @kaltimkece</h2>
<img src="%s"></img>
</body>
</html>"""

whole = message % (img5, img4, img1, img2, img3, img6, img7)
f.write(whole)
f.close()

webbrowser.open_new_tab('kaltimkece.html')