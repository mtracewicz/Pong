import numpy as np


class NeuralNetwork():
    def __init__(self, neurons_per_layer=np.array([4, 5, 9, 1]), weights=None):
        self._neurons_per_layer = neurons_per_layer
        self.weights = []
        self.biases = []
        for i in range(len(neurons_per_layer)-1):
            self.weights.append(np.random.randn(
                neurons_per_layer[i+1], neurons_per_layer[i]))
            self.biases.append(np.random.randn(neurons_per_layer[i+1], 1))

    def derived_relu(self, x):
        y = np.copy(x)
        y[y >= 0] = 1
        y[y < 0] = 0
        return y

    def relu(self, x):
        y = np.copy(x)
        y[y < 0] = 0
        return y

    def backpropagation(self, y, z_s, a_s):
        dw = []
        db = []
        deltas = [None] * len(self.weights)
        deltas[-1] = ((y-a_s[-1]) * (self.derived_relu(z_s[-1])))
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(
                self.derived_relu(z_s[i]))
            batch_size = y.shape[1]
            db = [d.dot(np.ones((batch_size, 1)))/float(batch_size)
                  for d in deltas]
            dw = [d.dot(a_s[i].T)/float(batch_size)
                  for i, d in enumerate(deltas)]
            return dw, db

    def fit(self, data, epochs=20, learning_rate=0.05):
        batch_size = data.shape[0]//20 + 1
        x = data[:, :4]
        y = data[:, 4]
        for _ in range(epochs):
            i = 0
            while(i < len(y)):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                i = i+batch_size
                z_s, a_s = self.predict(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [w+learning_rate*dweight for w,
                                dweight in zip(self.weights, dw)]
                self.biases = [w+learning_rate*dbias for w,
                               dbias in zip(self.biases, db)]

    def predict(self, x):
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            z_s.append(np.sum(self.weights[i].dot(a) + self.biases[i], axis=1))
            a = self.relu(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)

    def save_model(self):
        with open('model.txt', 'w') as f:
            f.write(str(self._neurons_per_layer))
            f.write('#####')
            f.write(str(self._weights))
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
