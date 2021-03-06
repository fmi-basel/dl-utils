from dlutils.training.lr_finder import lr_finder

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

import numpy as np


def create_model():
    '''
    '''
    model = Sequential()
    model.add(
        Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())

    for _ in range(2):
        model.add(Convolution2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def compile(model):
    '''
    '''
    model.compile(optimizer='sgd', loss='categorical_crossentropy')


def test_lr_finder(steps=100):
    '''
    '''
    # prepare dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., None]
    y_train = to_categorical(y_train)

    np.random.seed(13)

    def mnist_generator(batch_size):
        while True:
            idx = np.random.choice(len(x_train),
                                   size=batch_size,
                                   replace=False)
            yield x_train[idx, ...], y_train[idx, ...]

    batch_size = 10
    reps = 1

    for _ in range(reps):
        model = create_model()
        compile(model)
        lrf = lr_finder(model,
                        mnist_generator(batch_size),
                        steps=steps,
                        base_lr=1e-5,
                        max_lr=1.0,
                        verbose=1)

        assert len(lrf.lr) == steps
        assert len(lrf.losses) == 1
        assert len(lrf.losses[0]) == steps
        assert lrf.lr[0] == 1e-5
        assert lrf.lr[-1] == 1.0


if __name__ == '__main__':
    test_lr_finder()
