# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:29:02 2018

@author: yong
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
os.chdir(r'C:\Users\yong\Desktop\stat542\project4')

all=pd.read_csv('data.tsv', sep=' ', quotechar='"', escapechar='\\')
splits = pd.read_csv("splits.csv",low_memory=True, sep='\t')
s=3
train=all[all['new_id'].isin(splits.iloc[:,s-1])].reset_index(drop = True)
test=all[~all["new_id"].isin(splits.iloc[:,s-1])].reset_index(drop = True)

train.columns.values
all.columns.values

from bs4 import BeautifulSoup               # Split into words
import nltk
nltk.download()
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))

#Porter Stemming and Lemmatizing 

def review_to_word(raw_review,remove_stopwords=False):
    #remove HTML
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    #remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    words=[lemmatizer.lemmatize(token) for token in words]
    words=[lemmatizer.lemmatize(token, "v") for token in words]
    #so convert the stop words to a set  
    #optionally Remove stop words
    if remove_stopwords:
        words = [w for w in words if not w in stops]
    #Join the words back into one string separated by space, 
    # and return the result
    return( " ".join( words ))


#logistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
tfv.fit(clean_reviews)
train_data_features = tfv.fit_transform(clean_reviews)
vocal=tfv.get_feature_names()


model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
np.mean(cross_validation.cross_val_score(model, train_data_features, y, cv=20, scoring='roc_auc'))

model.fit(train_data_features,test["review"])
result = model.predict_proba(test_data_features)[:,1]
metrics.roc_auc_score(test["sentiment"], result)








num_reviews = train["review"].size
clean_reviews = []
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_reviews.append( review_to_word( train["review"][i],remove_stopwords=True ) )

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 

train_data_features = vectorizer.fit_transform(clean_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

#random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500) 
forest = forest.fit( train_data_features, train["sentiment"] )

num_reviews = len(test["review"])
clean_test_reviews = [] 
for i in range(0,num_reviews):
    clean_review = review_to_word( test["review"][i] )
    clean_test_reviews.append( clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame( data={"new_id":test["new_id"], "sentiment":result} )
probability = forest.predict_proba(test_data_features)[:,1]


from sklearn import metrics
metrics.roc_auc_score(test["sentiment"], probability)
#0.92142988889994626
#0.92783834551349542 500


#
model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
np.mean(cross_validation.cross_val_score(model, train_data_features, y, cv=20, scoring='roc_auc'))

model.fit(train_data_features,test["review"])
result = model.predict_proba(test_data_features)[:,1]
metrics.roc_auc_score(test["sentiment"], result)







 