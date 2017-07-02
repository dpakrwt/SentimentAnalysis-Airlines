#airlines sentiment anaysis
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import re
from html.parser import HTMLParser
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk import word_tokenize, pos_tag
from wordsegment import segment # it is used to break attached words

#Importing the airlines dataset
dataset = pd.read_csv('G:\\Data Science\\working_dir\\Airline-Sentiment-2-w-AA.txt',delimiter='\t', encoding ="ISO-8859-1")

html_parser = HTMLParser()

def removeReptitiveChars(word) :
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", word)


def removePunctuations(word) :
    exclude = set(string.punctuation)
    out = "".join(char for char in word if char not in exclude)
    return out
 

def apostropheLookUp(tweet) :
    APPOSTOPHES = {"'s" : " is", "'re" : " are", "'d" : " would", "'ll" : " will", "'ve" : " have", "'m" : " am", "'t" : " not"}
    words = tweet.split(' ')   
    modifiedTweet = ''
    for w in words :
        apostrophe = False
        for a in APPOSTOPHES :
            if a in w :
                apostrophe = True
                if a in "n't" :
                    reformed = w.split("'")[0][:-1] + APPOSTOPHES[a]
                else:
                    reformed = w.split("'")[0] + APPOSTOPHES[a]
       
        if(apostrophe == True) :
            modifiedTweet = modifiedTweet + ' ' + reformed
        else :
            modifiedTweet = modifiedTweet + ' ' + w
   
    return modifiedTweet.strip(" ")


def splitAttachedWords(tweet) :   
    cleaned = " ".join(re.findall('[A-Z][^A-Z]*', tweet))   
    return cleaned

def decodeData(tweet) :
    tweet = tweet.encode().decode('ascii', 'ignore')
    return tweet

def emoticonsToText(tweet) :
    emo_repl = {
		# positive emoticons
		":>": " good ",
		":d": " good ",
		":ddd": " good ",
		"=)": " happy ",
		"8)": " happy ",
		":-)": " happy ",
		":)": " happy ",
		";)": " happy ",
		"(-:": " happy ",
		"(:": " happy ",
		"=]": " happy ",
		"[=": " happy ",

		# negative emoticons
		":<": " sad ",
		":')": " sad ",
		":-(": " bad ",
		":(": " bad ",
		":S": " bad ",
		":-S": " bad ",
    } 

    emo_repl_order = [k for (k_len,k) in reversed(sorted([(len(k),k) for k in emo_repl.keys()]))]
    
    for k in emo_repl_order :
        tweet = tweet.replace(k, emo_repl[k])
    
    return tweet

#This method is use to filter out words that are not Nouns, Adjectives, Adverbs and Verbs from the tweets
def filterNonUsefulWords(tweet) :
    tagged = pos_tag(word_tokenize(tweet))
    filteredTweet = ''
    
    for i in range(0, len(tagged)) :
        word = tagged[i][0]
        tag = tagged[i][1]
                
        if tag.startswith('N') or tag.startswith('V') or tag.startswith('J') or tag.startswith('R') :
            filteredTweet = filteredTweet + ' ' + word
    
    return filteredTweet.strip()


#def spellCheck(word) :
                  
def preProcessTweet(tweet) :
 #   processedTweets = []
    #pre-processing the tweets
    #1) Convert each tweet to lower case
    tweet = tweets[i].lower()
       
    #2) Convert www.* or http?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
       
    #3) Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
       
    #3) Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
      
    #4) Replace #hastags with hashtag only
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
      
    #5) trim
    tweet = tweet.strip('\'"')
      
    #6) Escaping HTML tags
    tweet = html_parser.unescape(tweet)
    
    #7) Emoticons to Text
    tweet = emoticonsToText(tweet)
       
    #7) Splitting attached words
    tweet = ' '.join(word for word in segment(tweet))
    
    #8) Apostrophe lookup
    tweet = apostropheLookUp(tweet)
      
    #9) Decoding data
    tweet = decodeData(tweet)
      
    #10) slanglookup
    #tweet = _slang_lookup(tweet)
      
    #Split the tweets to further clean word by word
    words = tweet.split()
      
    cleanedTweet = ''
    for w in words:
        #replace two or more with two occurences
        w = removeReptitiveChars(w)
        
        #Strip punctuations attached to the word
        w = removePunctuations(w)
        
        #Check if the word start with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
      
        #skipping the stopwords and stemming
        lem = WordNetLemmatizer()
        
        if not w in set(stopwords.words('english') or val is not None) :
            word = lem.lemmatize(w)
            cleanedTweet = cleanedTweet + " " + word
        else:
            continue

#10) Filter out un-useful words
    cleanedTweet = filterNonUsefulWords(cleanedTweet)
        
    return cleanedTweet.strip()
#    corpus.append(cleanedTweet.strip(" "))
#    return corpus


#fetching the required columns
tweets = dataset['text'].values
sentiment = dataset['airline_sentiment'].values

corpus = []
for i in range(0, len(tweets)) :
    print(preProcessTweet(tweets[i]))
    corpus.append(preProcessTweet(tweets[i]))

#print(sentiment)
#creating bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
x = tfid.fit_transform(preProcessTweets(tweets)).toarray()

#print(sentiment)
                                
sentimentVector = []

for i in range(0, len(sentiment)):
    if sentiment[i] == 'positive' :
        v = 1
    elif sentiment[i] == 'negative' :
        v = -1
    else:
        v = 0
    sentimentVector.append(v)

#print(sentimentVector)

y = sentimentVector


#Splitting into test set and train set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(x_train, y_train)

# Predicting the test set results
y_pred1 = classifier1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)


#Fitting SVM 
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf')
classifier2.fit(x_train, y_train)

y_pred2 = classifier2.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred2)