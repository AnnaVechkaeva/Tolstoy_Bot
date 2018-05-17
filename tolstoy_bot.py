# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
import tweepy
from time import sleep

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

text = ""
with open("corpus.txt", "r+", encoding="utf-8") as f:
    for line in f:
        text += (line.rstrip('\n')).lower()
f.close()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 10
step = 60
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])   

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))


optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam')

twitts = []
for iteration in range(1, 4):    
    print('\n\nIteration', iteration)
    history = model.fit(X, y, batch_size=40, nb_epoch=10)
    for i  in range(10):

        start_index = random.randint(0, len(text) - maxlen - 1)

        diversity = 1.0
        print('\nDiversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        sys.stdout.write(generated)
        len_tw = random.randint(1, 120)
        for i in range(len_tw):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, 1.0)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        twitts.append(generated)

    loss = (history.history['loss'])
    perplexity = [math.pow(2, l) for l in loss]
    
    plt.plot(perplexity)
    plt.title('Model Perplexity -- Iteration ' + str(iteration))
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.savefig('iteration' + str(iteration) + '.png')
    plt.clf()

scriptpath = "./credentials.py"
sys.path.append(os.path.abspath(scriptpath))
from credentials import *
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
for twitt in twitts:
    print(line)
    api.update_status(line)
    sleep(600)
