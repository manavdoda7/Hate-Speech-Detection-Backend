import pandas as panda
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
dataset = panda.read_csv("./data3.csv")
dataset = dataset.sample(frac=1)
print(dataset)
X = dataset['tweet'].astype(str)
y = dataset['label'].astype(int)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxLen = 120
training_samples = 1400000
validation_samples = 17182
max_words = 250000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
print('Tokenizer words: ', len(word_index))

data = pad_sequences(sequences, maxlen = maxLen)
labels = np.asarray(y)
print(data.shape)
print(labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.2)

from nltk.stem.porter import *
glove_dir = './glove_embeddings_100d.txt'
stemmer = PorterStemmer()
embeddings_index = {}
f = open(glove_dir, encoding = 'utf8')
for line in f:
  values = line.split()
  word = stemmer.stem(values[0])
  coefs = np.asarray(values[1:], dtype = 'float32')
  embeddings_index[word] = coefs
f.close()

print(len(embeddings_index))

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if i<max_words:
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

import tensorflow as tf
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), 
                                 initializer='glorot_uniform', 
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)
        
    def call(self, inputs):
        score = tf.math.tanh(tf.matmul(inputs, self.W) + self.b)
        score = tf.matmul(score, tf.expand_dims(self.u, axis=1))
        score = tf.squeeze(score, axis=-1)
        weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(tf.transpose(inputs, [0, 2, 1]), 
                                   tf.expand_dims(weights, axis=-1))
        context_vector = tf.squeeze(context_vector, axis=-1)
        return context_vector
    
# import torch
# t = tf.convert_to_tensor(X)
import tensorflow as tf

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU, using {}.".format(device_name))

with tf.device(device_name):
    model = tf.keras.Sequential([
        
        # Embedding layer
        tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxLen),
        
        # Convolutional layer
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),


        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, return_sequences=True)),
        Attention(),
        
        tf.keras.layers.Flatten(),

        # First dense layer
        tf.keras.layers.Dense(units=64, activation='relu'),
        
        # Second dense layer
        tf.keras.layers.Dense(units=32, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(
        optimizer='rmsprop',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
        run_eagerly=True
    )

model.fit(X_train, y_train, epochs=3, batch_size=32)

# Evaluate model
model.evaluate(X_test, y_test)

# Use model to make predictions
predictions = model.predict(X_test)

filename = './GRU_Model/GRU_model_weights.h5'
model.save_weights(filename)

import pickle
filename = './GRU_Model/GRU_Model.pkl'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(X_train, open('./GRU_Model/X_train.pkl', 'wb'))
pickle.dump(X_test, open('./GRU_Model/X_test.pkl', 'wb'))
pickle.dump(y_train, open('./GRU_Model/Y_train.pkl', 'wb'))
pickle.dump(y_test, open('./GRU_Model/Y_test.pkl', 'wb'))
pickle.dump(tokenizer, open('./GRU_Model/tokenizer.pkl', 'wb'))