import numpy as np
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import KFold
from keras.layers import BatchNormalization
from sklearn.metrics import roc_auc_score, auc
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def model_keras():
    print('model_keras is running...')

    dropout = 0.4

    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(units=250, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('model_keras is done.')
    return model
