import os


import numpy as np


class NeuralNetwork():
    def __init__(self, neurons_per_layer=np.array([4, 2, 2, 1]), weights=None, biases=None):
        if weights is None or biases is None:
            self.weights = []
            self.biases = []
            for i in range(len(neurons_per_layer)-1):
                self.weights.append(np.random.randn(
                    neurons_per_layer[i+1], neurons_per_layer[i]))
                self.biases.append(np.random.randn(
                    neurons_per_layer[i+1]))
        else:
            self.weights = weights
            self.biases = biases

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def derived_sigmoid(self, x):
        return x * (1 - x)

    def backpropagation(self, weights, y, activated):
        delta_weights = []
        delta_biases = []
        deltas = [None] * len(weights)
        deltas[-1] = ((y-activated[-1]) *
                      (self.derived_sigmoid(activated[-1])))
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = weights[i+1].T.dot(deltas[i+1]) * \
                (self.derived_sigmoid(activated[i+1]))
        delta_biases = deltas
        delta_weights = [d.dot(activated[i+1].T)
                         for i, d in enumerate(deltas)]
        return delta_weights, delta_biases

    def fit(self, data, epochs=25, learning_rate=0.1):
        x = data[:, : 4]
        y = data[:, 4]
        for _ in range(epochs):
            i = 0
            while(i < x.shape[0] - 1):
                i += 1
                activated = self.predict(x[i])
                delta_weights, delta_biases = self.backpropagation(
                    self.weights, y[i], activated)
                self.weights = [w+learning_rate*dweight for w,
                                dweight in zip(self.weights, delta_weights)]
                self.biases = [w+learning_rate*dbias for w,
                               dbias in zip(self.biases, delta_biases)]

    def predict(self, x):
        a = np.copy(x)
        pre_activation = []
        activated = [a]
        for i in range(len(self.weights)):
            pre_activation.append(self.weights[i].dot(a) + self.biases[i])
            a = self.sigmoid(pre_activation[-1])
            activated.append(a)
        return activated

    def save_model(self):
        np.save(os.path.join("model", "weights.npy"), self.weights)
        np.save(os.path.join("model", "biases.npy"), self.biases)

    @ staticmethod
    def load_model():
        weights = np.load(os.path.join("model", "weights.npy"))
        biases = np.load(os.path.join("model", "biases.npy"))
        return NeuralNetwork(weights=weights, biases=biases)
