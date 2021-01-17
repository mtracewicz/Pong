import numpy as np


class NeuralNetwork():
    def __init__(self, neurons_per_layer=np.array([4, 2, 2, 1]), weights=None):
        self._neurons_per_layer = neurons_per_layer
        self.weights = []
        self.biases = []
        for i in range(len(neurons_per_layer)-1):
            self.weights.append(np.random.randn(
                neurons_per_layer[i+1], neurons_per_layer[i]))
            self.biases.append(np.random.randn(
                neurons_per_layer[i+1]))
        self.learning = False

    def relu(self, x):
        y = np.copy(x)
        y[y < 0] = 0
        return y

    def derived_relu(self, x):
        y = np.copy(x)
        y[y >= 0] = 1
        y[y < 0] = 0
        return y

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
            deltas[i] = weights[i+1].T.dot(deltas[i+1])
            deltas[i] *= (self.derived_sigmoid(activated[i+1]))
        delta_biases = deltas
        delta_weights = [d.dot(activated[i+1].T)
                         for i, d in enumerate(deltas)]
        return delta_weights, delta_biases

    def fit(self, data, epochs=5, learning_rate=0.05):
        self.learning = True
        w, b = self.internal_fit(data, epochs, learning_rate)
        self.learning = False
        self.weights = np.copy(w)
        self.biases = np.copy(b)

    def internal_fit(self, data, epochs=20, learning_rate=0.05):
        x = data[:, : 4]
        y = data[:, 4]
        _weights = np.copy(self.weights)
        _biases = np.copy(self.biases)
        for _ in range(epochs):
            i = 0
            while(i < x.shape[0] - 1):
                i += 1
                activated = self.predict(
                    _weights, _biases, x[i])
                delta_weights, delta_biases = self.backpropagation(
                    _weights, y[i], activated)
                _weights = [w+learning_rate*dweight for w,
                            dweight in zip(_weights, delta_weights)]
                _biases = [w+learning_rate*dbias for w,
                           dbias in zip(_biases, delta_biases)]
        return (_weights, _biases)

    def predict(self, weights, biases, x):
        a = np.copy(x)
        pre_activation = []
        activated = [a]
        for i in range(len(weights)):
            pre_activation.append(weights[i].dot(a) + biases[i])
            a = self.sigmoid(pre_activation[-1])
            activated.append(a)
        return activated

    def make_predict(self, x):
        a = np.copy(x)
        pre_activation = []
        activated = [a]
        for i in range(len(self.weights)):
            pre_activation.append(self.weights[i].dot(a) + self.biases[i])
            a = self.sigmoid(pre_activation[-1])
            activated.append(a)
        return activated

    def save_model(self):
        with open('model.txt', 'w') as f:
            f.write(str(self._neurons_per_layer))
            f.write('#####')
            f.write(str(self. weights))
            f.flush()

    @staticmethod
    def load_model():
        with open('model.txt', 'r') as f:
            text = f.read()
            text = text.split('#####')
        tmp = (list((text[0][1:-1]).split(' ')))
        neurons_per_layer = [i[:-1] if i[-1] == ',' else i for i in tmp]
        neurons_per_layer = np.array(neurons_per_layer, dtype=np.uint8)
        tmp = (list((text[1][1:-1].replace('\n', '')).split(' ')))
        weights = [i for i in tmp if i != '']
        weights = np.array(weights, dtype=np.float)
        return NeuralNetwork(neurons_per_layer, weights)
