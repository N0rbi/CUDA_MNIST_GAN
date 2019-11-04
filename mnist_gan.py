import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
np.warnings.filterwarnings('ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils

from sklearn.utils import shuffle

IM_SIZE = 28*28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], IM_SIZE).astype(float)
X_test = X_test.reshape(X_test.shape[0], IM_SIZE).astype(float)

X_train /= 255.0
X_test /= 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

generator_in = Input(shape=(10,))
generator = Dense(256, activation='relu')(generator_in)
generator = Dense(256, activation='relu')(generator)
generator = Dense(512, activation='relu')(generator)
generator = Dense(512, activation='relu')(generator)
generator = Dense(IM_SIZE, activation='sigmoid')(generator)

generator_model = Model(generator_in, generator)

generator_model.compile(loss='MSE',
                optimizer='SGD',
                metrics=['accuracy'])


discriminator_in = Input(shape=(IM_SIZE,))
discriminator = Dense(256, activation='relu')(discriminator_in)
discriminator = Dense(256, activation='relu')(discriminator)
discriminator = Dense(256, activation='relu')(discriminator)
discriminator = Dense(1, activation='sigmoid')(discriminator)

discriminator_model = Model(discriminator_in, discriminator)

discriminator_model.compile(loss='binary_crossentropy',
                optimizer='SGD',
                metrics=['accuracy'])

gan_input = Input(shape=(10,))
GAN = generator_model(gan_input)
GAN = discriminator_model(GAN)
GAN_model = Model(gan_input, GAN)

X = np.random.uniform(0.0, 1.0, size=X_train.shape)

X = np.concatenate([X,X_train])

Y = np.concatenate([np.zeros(60000), np.ones(60000)])

X, Y = shuffle(X,Y)

discriminator_model.fit(X, Y, epochs=1, verbose=1)

X_noise = np.random.uniform(0.0, 1.0, size=(60000, 10))
Y_noise = np.ones(60000)

make_trainable(discriminator_model, False)

GAN_model.compile(loss='binary_crossentropy',
                optimizer='SGD',
                metrics=['accuracy'])


GAN_model.fit(X_noise, Y_noise, epochs=20)

print("Score on test: %f" % eval)


from matplotlib import pyplot as plt
plt.imshow(img, interpolation='nearest')
plt.show()