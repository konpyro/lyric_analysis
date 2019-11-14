from __future__ import print_function

import csv
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
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
        count += 1

# stopwords removal


def thorough_filter(words):
    filtered_words = []
    for word in words:
        if len(word) == 1 or word == "'s":  # removes one letter words and "'s" suffix
            continue
        pun = []
        for letter in word:
            pun.append(letter in string.punctuation)
        if not all(pun):
            filtered_words.append(word)
    return filtered_words


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

filtered_X = []
thorough_filtered_X = []
lem_X = []
stem_X = []
for i in range(len(X)):
    X[i] = X[i].lower()
    X[i] = nltk.word_tokenize(X[i])
    filtered_X.append([word for word in X[i] if word not in stopwords.words('english') + list(string.punctuation)])
    thorough_filtered_X.append(thorough_filter(filtered_X[i]))
    #    print(len(X[i]), len(filtered_X[i]), len(thorough_filtered_X[i]))
    #    print(X[i][:10], '\n', filtered_X[i][:10], '\n', thorough_filtered_X[i][:10])

# lemmatization
    lem_X.append([lemmatizer.lemmatize(word) for word in thorough_filtered_X[i]])

# stemming
    stem_X.append([stemmer.stem(word) for word in thorough_filtered_X[i]])
    #    print(thorough_filtered_X[i][:10], '\n', lem_X[i][:10], '\n', stem_X[i][:10])

# output
y_t = np.zeros((len(Y), 4))
for i  in range(len(Y)):
    if Y[i] == 'happy':
        y_t[i][:] = [1, 0, 0, 0]
    elif Y[i] == 'angry':
        y_t[i][:] = [0, 1, 0, 0]
    elif Y[i] == 'sad':
        y_t[i][:] = [0, 0, 1, 0]
    elif Y[i] == 'relaxed':
        y_t[i][:] = [0, 0, 0, 1]
y_train = y_t[:150][:]
y_test = y_t[150:200][:]


# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(thorough_filtered_X)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
input_bow = t.texts_to_matrix(thorough_filtered_X, mode='count')  #  mode='count' for BOW and mode='tfidf' for TFIDF
input_tfidf = t.texts_to_matrix(thorough_filtered_X, mode='tfidf')

# input to Keras renaming
x_train = input_bow[:150][:]
x_test = input_bow[150:200][:]


print('Build model...')
model = Sequential()
model.add(Embedding(5000, 64))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          epochs=1,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)
# add validation