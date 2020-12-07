from random import randint
import numpy as np

class NeuralNetwork():
    def __init__(self, neurons_per_layer = np.array([5,4]), weights = None):
        self._hidden_layers = len(neurons_per_layer)-1
        self._neurons_per_layer = neurons_per_layer
        number_of_weights = int(np.sum(np.array([self._neurons_per_layer[i]*self._neurons_per_layer[i+1] for i in range(self._hidden_layers)]))+self._neurons_per_layer[-1])
        self._weights = weights or np.random.rand(number_of_weights) - 0.5

    def fit(self, inputs, outputs, learning_rate = 0.05):
        pass

    def predict(self, inputs):
        return randint(-1,2)
        # return (int(abs(np.sum(inputs*self._weights)))%3) -1;

    def save_model(self):
        with open('model.txt','w') as f:
            f.write(np.array2string(self._neurons_per_layer))
            f.write('#####')
            f.write(np.array2string(self._weights))
            f.flush()

    @staticmethod
    def load_model():
        with open('model.txt','r') as f:
            text = f.read()
            text = text.split('#####')
        neurons_per_layer = np.fromstring(text[0])
        weights = np.fromstring(text[1]).reshape(neurons_per_layer)
        return NeuralNetwork(neurons_per_layer, weights)