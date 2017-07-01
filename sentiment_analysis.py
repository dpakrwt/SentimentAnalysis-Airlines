import nltk
import numpy as np
nltk.download('sentiwordnet')

from nltk import word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn


tweets = [
"New story to Tamil industry Very good try Congratulations to the director Better to try to new concept rather than saying same old concept again and again songs are good",
"Time pass Movie not worth to watch in theater",
"super flop movie",
"It is worst movie to watch",
"Amazing movie Salman acting always good must watch it"
]


#Cleaning and pre-processing tweets-

featureVector = []
sentiments = []

for i in range (0, len(tweets)) :
    positive_score = negative_score = objective_score = count = 0
    tokens = word_tokenize(tweets[i].lower())
    tagged = pos_tag(tokens)
    
    for j in range (0, len(tagged)) :
        scores = None
        tag = tagged[j][1]
        word = tagged[j][0]
        
        if 'NN' in tag and swn.senti_synsets(word):
            scores = list(swn.senti_synsets(word))
        elif 'VB' in tag and swn.senti_synsets(word):
            scores = list(swn.senti_synsets(word))
        elif 'JJ' in tag and swn.senti_synsets(word):
            scores = list(swn.senti_synsets(word))
        elif 'RB' in tag and swn.senti_synsets(word):
            scores = list(swn.senti_synsets(word))
        else:
            continue
                
        if scores:    
            positive_score = positive_score + scores[0].pos_score()
            negative_score = negative_score + scores[0].neg_score()
            objective_score = objective_score + scores[0].obj_score()
        count +=1
        
    print("positive_score = ", positive_score)
    print("negative score = ", negative_score) 
    print("objective score = ", objective_score)
    
    final_score=positive_score - negative_score
    print("final_Score", final_score)
    
    norm_finalscore= round((final_score) / count, 2)
    
    print("norm_finalscore", norm_finalscore)
    
    final_sentiment = 'positive' if norm_finalscore >= 0 else 'negative'
    
    print(tweets[i], "----->", final_sentiment)
    
    featureVector.append(tweets[i])
    sentiments.append(final_sentiment)
    

#print(featureVector)
sentimentVector = []

for i in range(0, len(sentiments)):
    if sentiments[i] == 'positive' :
        v = 1
    else:
        v = 0
    sentimentVector.append(v)

print(sentimentVector)


from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
featureSpace = tfid.fit_transform(featureVector)
print(featureSpace)

#splitting dataset into training and test sets
#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(featureSpace.toarray(), sentimentVector)

test_review = ['a must not watch movie']

test_features = tfid.transform(test_review)

print(test_features.toarray())

y_pred1 = classifier1.predict(test_features.toarray())

print("Prediction of Naive Bayes = " , y_pred1)


#Fitting SVM 
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf')
classifier2.fit(featureSpace.toarray(), sentimentVector)

y_pred2 = classifier2.predict(test_features.toarray())

print("Prediction of SVM = ", y_pred2)



from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression(random_state=0)
classifier3.fit(featureSpace.toarray(), sentimentVector)

y_pred3 = classifier3.predict(test_features.toarray())

print("Prediction of Logistic Regression = ", y_pred3)