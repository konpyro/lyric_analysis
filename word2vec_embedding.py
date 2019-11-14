from __future__ import print_function

import csv
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('rslp')

# X: input, Y:output
X = []
Y = []




with open('MoodyLyricsFull4Q.csv') as file:
    reader = csv.reader(file, delimiter = ',')
    count = 0
    for row in reader:
        if count > 0:
            X.append(row[4])
            Y.append(row[3])

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

lyric_lines = list()
lines = X

for line in lines:
    tokens = word_tokenize(line)

    tokens = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    lyric_lines.append(words)


import gensim

model = gensim.models.Word2Vec(sentences=lyric_lines, size=100, window=5, workers=4, min_count=1)

words = list(model.wv.vocab)
print("Vocabulary size: %d" %len(words))

filename = 'lyric_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary='False')
