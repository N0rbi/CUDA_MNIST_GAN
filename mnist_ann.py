import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
np.warnings.filterwarnings('ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

IM_SIZE = 28*28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], IM_SIZE).astype(float)
X_test = X_test.reshape(X_test.shape[0], IM_SIZE).astype(float)

X_train /= 255.0
X_test /= 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential([Dense(256, input_shape=(IM_SIZE,), activation='relu'), 
                    Dense(256, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(10, activation='softmax')])

model.compile(loss='categorical_crossentropy',
                optimizer='SGD',
                metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

eval = np.mean(model.evaluate(X_test, y_test, verbose=1))

print("Score on test: %f" % eval)