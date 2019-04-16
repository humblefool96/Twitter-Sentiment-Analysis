import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = pd.concat([train, test], ignore_index = True)

#data inspection
"""print(train[train['label'] == 0].head(10))
print(train[train['label'] == 1].head(10))
print(train.shape, test.shape, data.shape)
print(train['label'].value_counts())
print(data['label'].value_counts())

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.hist(length_train, bins = 20, label = 'train_tweets')
plt.hist(length_test, bins = 20, label = 'test_tweets')
plt.legend()
plt.show()"""

#data cleaning
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
corpus = []
for i in range(0, 49159):
    tweet = re.sub('[^a-zA-Z#]', ' ', data['tweet'][i])
    tweet = tweet.lower()
    #tweet = tweet.split()
    tokenize_tweet = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    #ps = PorterStemmer()
    #tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    #tweet = ' '.join(tweet)
    filtered_tweet = []
    for word in tokenize_tweet :
        if word not in stop_words:
            filtered_tweet.append(word)
            lem.lemmatize(word)
    filtered_tweet = ' '.join(filtered_tweet)
    corpus.append(filtered_tweet)
    
#creating the bag of words model    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 25000)
bow = cv.fit_transform(corpus).toarray()
train_bow = bow[:31962, :] 
test_bow = bow[31962:, :]
y_bow = data.iloc[:31962, 1].values

#splitting the dataset
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_bow, y_bow, test_size = 0.20)"""
   
# Fitting Naive Bayes to the Training set
"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
classifier.fit(train_bow, y_bow)"""

"""from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_bow, y_bow)

# Predicting the Test set results
#y_pred = classifier.predict(x_test)
y_pred = classifier.predict(test_bow)

# Making the Confusion Matrix
"""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)"""   

submission=pd.read_csv("sample.csv")
submission['label']=y_pred
pd.DataFrame(submission, columns=['id','label']).to_csv('result.csv')