# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:22:26 2023

@author: oberdan.pinheiro
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset = pd.read_csv('test_all.csv', encoding = 'ISO-8859-1')

dataset = dataset.sample(frac = 1)

train_dataset = dataset.sample(frac=0.70,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("class")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('class')
test_labels = test_dataset.pop('class')

labelencoder = LabelEncoder()
train_labels = labelencoder.fit_transform(train_labels)
train_labels = np_utils.to_categorical(train_labels)

test_labels = labelencoder.fit_transform(test_labels)
test_labels = np_utils.to_categorical(test_labels)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

model = Sequential()
model.add(Dense(units = 128, activation = 'relu', input_dim = 5))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 4, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                  metrics = ['categorical_accuracy'])

model.summary()

model.fit(
    normed_train_data, train_labels,
    epochs=1000,
    batch_size=64
)

score = model.evaluate(normed_test_data, test_labels)

test_predictions = model.predict(normed_test_data)

class_predict = [np.argmax(t) for t in test_labels]
predict = [np.argmax(t) for t in test_predictions]

matriz = confusion_matrix(predict, class_predict)

print(matriz)

labels = [0, 1, 2, 3]
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()