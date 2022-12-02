# import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from contextlib import redirect_stdout


def model_keras():

    batch_size = 64
    dropout = 0.5
    kernel_size = (5, 5)
    leakyReLU = LeakyReLU(alpha=0.3)
    epoch = 10

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=kernel_size, activation=leakyReLU, padding='same', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=kernel_size, activation=leakyReLU, padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(units=1024, activation=leakyReLU))
    model.add(Dropout(dropout))
    model.add(Dense(units=16, activation=leakyReLU))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))

    with open('Model_Summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model