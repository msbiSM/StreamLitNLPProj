# Commented out IPython magic to ensure Python compatibility.
# Importing necessary libraries for data loading and exploration


#from flask import Flask, jsonify, request
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

import re
import string
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
snowball = SnowballStemmer(language='english')

import torch
import torchvision

import time

import nlpaug.augmenter.word.context_word_embs as aug

augmenter = aug.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

import joblib
import streamlit as st

#from lime.lime_text import LimeTextExplainer
#import streamlit.components.v1 as components

from pandasql import sqldf
from wordcloud import WordCloud
###################################################

# https://stackoverflow.com/a/47091490/4084039


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
# <br /><br /> ==> after the above steps, we are getting "br br"
# we are including them into stop words list
# instead of <br /> if we have <br/> these tags would have revmoved in the 1st step

stopwords = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])



#function to create new field using the rating column

def gettargetlabel(rating):
  if rating in (1,2):
    return 0
  elif rating in (4,5):
    return 1
  #else:
    #return -1

#Text Pre-processing review texts with stemming
def text_preprocess_label(text):
  #preprocessed_reviews = []
  # tqdm is for printing the status bar
  #remove urls if any
  text = re.sub(r'http\S+','',text)
  #remove numbers if any
  text = re.sub(r'\d+', '', text)
  #remove punctuations
  text = [c for c in text if c not in string.punctuation]
  text = ''.join(text)
  #remove whitespaces if any
  text = ' '.join(text.split())
  #convert text to lowercase
  text = text.lower()
  #convert decontracted words
  text = decontracted(text)
  #remove stopwords
  clean_text = [word.lower() for word in text.split() if word.lower() not in stopwords]
  clean_text = ' '.join(clean_text)
  #print('Stop words output: ', clean_text)
  #remove duplicates
  remov_dup_text = re.sub(r'\b(\w+)(?:\W+\1\b)+',r'\1',clean_text)
  #print('Removed duplicates: ', remov_dup_text)
  #apply stemming
  stemmed_text = " ".join([snowball.stem(word) for word in remov_dup_text.split()])
  #preprocessed_reviews.append(stemmed_text.strip())
  return stemmed_text
  
def augmentMyData(df, augmenter, repetitions=1, samples=200):
    augmented_texts = []
    # select only the minority class samples
    neg_df = df[df['Label'] == 0].reset_index(drop=True) # removes unecessary index column
    for i in tqdm(np.random.randint(0, len(neg_df), samples)):
        # generating 'n_samples' augmented texts
        for _ in range(repetitions):
            augmented_text = augmenter.augment(neg_df['CompleteReview'].iloc[i])
            augmented_texts.append(augmented_text)
    
    data = {
        'Label': 0,
        'CompleteReview': augmented_texts
    }
    aug_df = pd.DataFrame(data)
    df = shuffle(df.append(aug_df).reset_index(drop=True))
    return df
    
def contostr(s):
  # initialize an empty string
  str1 = ""
 
  # traverse in the string
  for ele in s:
    str1 += ele
 
  # return string
  return str1

#Text Pre-processing review texts without stemming
def text_preprocess_wordcloud(text):
  preprocessed_reviews = []
  # tqdm is for printing the status bar
  for txt in tqdm(text.values):
    #remove urls if any
    txt = re.sub(r'http\S+','',txt)
    #remove numbers if any
    txt = re.sub(r'\d+', '', txt)
    #remove punctuations
    txt = [c for c in txt if c not in string.punctuation]
    txt = ''.join(txt)
    #remove whitespaces if any
    txt = ' '.join(txt.split())
    #convert text to lowercase
    txt = txt.lower()
    #convert decontracted words
    txt = decontracted(txt)
    #remove stopwords
    txt = [word.lower() for word in txt.split() if word.lower() not in stopwords]
    txt = ' '.join(txt)
    #print('Stop words output: ', clean_text)
    #remove duplicates
    txt = re.sub(r'\b(\w+)(?:\W+\1\b)+',r'\1',txt)
    #print('Removed duplicates: ', remov_dup_text)
    preprocessed_reviews.append(txt.strip())
  return preprocessed_reviews

###################################################

st.write("# Employee Review Text Classification")

message_text = st.text_input("Enter a message for review classification")

###################################################

clf = joblib.load('rf_model.pkl')
tfidf_vect = joblib.load('tfidf_vect.pkl')
companywisereviews = joblib.load('companywisereviews.pkl')
    
#prob_result = np.array([])
def convert_text_to_array(message):
    review_text = text_preprocess_label(message)
    test_vect = tfidf_vect.transform(([review_text]))
    X_test_tf = test_vect.toarray()
    
    return X_test_tf
    
def predict(model,message):
    X_test_tf = convert_text_to_array(message)
    pred = clf.predict(X_test_tf)
    if pred[0]>0.5:
        prediction = "Positive"
    else:
        prediction = "Negative"
    reviewtext_prob = clf.predict_proba(X_test_tf)
    neg_prob = reviewtext_prob[0][0]
    pos_prob = reviewtext_prob[0][1]
    

    return {'prediction': prediction, 'Review Text is Positive %': pos_prob*100, 'Review Text is Negative %':neg_prob*100}

#def predict_proba(message_text):
#    X_test_tf = convert_text_to_array(message_text)
#    #exp = explainer.explain_instance(message,clf.predict_proba, num_features=10)
#    pred_proba = clf.predict(X_test_tf)
    
#    #format_pred = np.concatenate([1.0-pred_proba, pred_proba], axis=1)
    
#    return pred_proba

distinctCompanies = sqldf(""" select distinct CompanyName from companywisereviews union select '' as CompanyName order by CompanyName """)

for_company = st.sidebar.selectbox('Select the Company for which review is being classified:', distinctCompanies)
    

if message_text != '' and for_company != '':
    result = predict(clf,message_text)
    
    st.write(result)
    
    labels = 'Positive Text', 'Negative Text'
    sizes = [result['Review Text is Positive %'],result['Review Text is Negative %']]
    explode = (0, 0.1)
    
    fig1,ax1 = plt.subplots()
    plt.figure(figsize=(10,6))
    ax1.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=False)
    ax1.axis('equal')
    
    st.pyplot(fig1)

    df_reviews = companywisereviews["CompleteReview"].loc[companywisereviews["CompanyName"] == for_company]

    #st.dataframe(df_reviews)
    reviews = text_preprocess_wordcloud(df_reviews) #applying text preprocesing on review titles
    reviewstitle_nodup = re.sub(r'\b(\w+)(?:\W+\1\b)+',r'\1',(','.join(reviews))) #removing the duplicate words

    st.set_option('deprecation.showPyplotGlobalUse', False)

    #creating wordcloud for the review title
    wordcloud = WordCloud(stopwords = stopwords,background_color="white").generate(reviewstitle_nodup)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()


    #explain_pred = st.button('Explain Predictions')
    
    #if explain_pred:
    #    with st.spinner('Generating explanations'):
    #        class_names = ['Positive', 'Negative']
    #        explainer = LimeTextExplainer(class_names=class_names)
    #        exp = explainer.explain_instance(message_text, predict_proba, num_features=10)
    #        components.html(exp.as_html(), height=800)

elif message_text != '' and for_company == '':
    result = predict(clf,message_text)
    
    st.write(result)
    
    labels = 'Positive Text', 'Negative Text'
    sizes = [result['Review Text is Positive %'],result['Review Text is Negative %']]
    explode = (0, 0.1)
    
    fig1,ax1 = plt.subplots()
    ax1.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=False)
    ax1.axis('equal')
    
    st.pyplot(fig1)
