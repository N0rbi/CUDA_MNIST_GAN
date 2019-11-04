from keras.datasets import mnist
import numpy as np

IM_SIZE = 28*28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], IM_SIZE).astype(float)
X_test = X_test.reshape(X_test.shape[0], IM_SIZE).astype(float)

X_train /= 255.0
X_test /= 255.0

def sigmoid(vec):
    return 1 / (1 + np.exp(vec))

def d_sigmoid(vec):
    return vec * (1 - vec)

def relu(vec):
    return np.maximum(0, vec)

def d_relu(vec):
    return np.maximum(0, vec) / np.maximum(0, vec) # 0 or 1

layers = [(IM_SIZE, 256, relu, d_relu), (256, 256, relu, d_relu), (256, 256, relu, d_relu), (256, 1, sigmoid, d_sigmoid)]

def glorot_init(fan_in, fan_out):
    variance = 2 / (fan_in + fan_out)
    stddev = np.sqrt(variance)
    return np.random.normal(0.0, stddev, (fan_in, fan_out))

def ce(gold, pred):
    return - np.sum(np.eye(10)[gold]*np.log(pred), axis=1)

weights = [glorot_init(l[0], l[1]) for l in layers]
biases = [np.zeros(l[1]) for l in layers]
activations = [l[2] for l in layers]
d_activations = [l[3] for l in layers]


for epoch in range(10):

    sample = X_train[0:1]
    before_values = []
    after_values = []
    for w, b, a in zip(weights, biases, activations):
        before_values.append(sample)
        sample = a((sample @ w) + b)
        after_values.append(sample)

    err = np.mean(ce(y_train[0:1], sample))
    update = err * -0.02
    for w, b, a, d_a, b_v, a_v in reversed(list(zip(weights, biases, activations, d_activations, before_values, after_values))):
        print(update.shape)
        print(d_a(a_v))
        update = update @ d_a(a_v)
        biases += update[0]
        update = b_v.T @ update
        w += update

    print(err)



