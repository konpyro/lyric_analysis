import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#X: input, Y:output
X = []
Y = []
with open('MoodyLyricsFullSmall4Q.csv') as file:
    reader = csv.reader(file, delimiter = ',')
    count = 0
    for row in reader:
        if count > 0:
            X.append(row[4])
            Y.append(row[3])
        count += 1

X_array = np.asarray(X).reshape(10,1)
Y_array = np.asarray(Y).reshape(10,1)


#Custom Tokenizers
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

#Custom Lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#Custom Stemmer

    def tokenize(text):
        # my text was unicode so I had to use the unicode-specific translate function. If your documents are strings, you will need to use a different `translate` function here. `Translated` here just does search-replace. See the trans_table: any matching character in the set is replaced with `None`
        tokens = [word for word in nltk.word_tokenize(text.translate(trans_table)) if len(word) > 1]  # if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
        stems = [stemmer.stem(item) for item in tokens]
        return stems



def StemTokenizer2(text):
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in text]
    return stems

class StemTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()
    def __call__(self, doc):
        return [self.ps.stem(t) for t in word_tokenize(doc)]

#Tf Idf Vectorizer
print("Dimensions before optimizing TfidfVectorizer parameters")
vectorizer = TfidfVectorizer()
tf_idf_array = vectorizer.fit_transform(X).toarray() # επιστρέφει sparse matrix, γι'αυτό το κάνουμε .toarray()
print('TF-IDF array shape:', tf_idf_array.shape)

print("Dimensions after optimizing TfidfVectorizer parameters")
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
tf_idf_array = vectorizer.fit_transform(X).toarray() # επιστρέφει sparse matrix, γι'αυτό το κάνουμε .toarray()
print('TF-IDF array shape:', tf_idf_array.shape)
"""""
print("Dimensions after optimizing TfidfVectorizer parameters, with Lemmatization")
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_df=0.5, min_df=2, stop_words='english')
tf_idf_array = vectorizer.fit_transform(X).toarray() # επιστρέφει sparse matrix, γι'αυτό το κάνουμε .toarray()
print('TF-IDF array shape:', tf_idf_array.shape)
"""
print("Dimensions after optimizing TfidfVectorizer parameters, with Stemming")
vectorizer = TfidfVectorizer(tokenizer=StemTokenizer2, max_df=0.5, min_df=2, stop_words='english')
tf_idf_array = vectorizer.fit_transform(X).toarray() # επιστρέφει sparse matrix, γι'αυτό το κάνουμε .toarray()
print('TF-IDF array shape:', tf_idf_array.shape)
