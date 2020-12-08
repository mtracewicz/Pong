from random import randint
import numpy as np

class NeuralNetwork():
    def __init__(self, neurons_per_layer = np.array([5,4]), weights = None):
        self._hidden_layers = len(neurons_per_layer)-1
        self._neurons_per_layer = neurons_per_layer
        number_of_weights = int(np.sum(np.array([self._neurons_per_layer[i]*self._neurons_per_layer[i+1] for i in range(self._hidden_layers)]))+self._neurons_per_layer[-1])
        self._weights = weights if weights is not None else np.random.rand(number_of_weights) - 0.5

    def fit(self, inputs, outputs, learning_rate = 0.05):
        pass

    def predict(self, inputs):
        #a = (int(np.sum(inputs*self._weights)))
        a = randint(-5,5)
        return -1 if a < 0 else 0 if a == 0 else 1

    def save_model(self):
        with open('model.txt','w') as f:
            f.write(str(self._neurons_per_layer))
            f.write('#####')
            f.write(str(self._weights))
            f.flush()

    @staticmethod
    def load_model():
        with open('model.txt','r') as f:
            text = f.read()
            text = text.split('#####')
        tmp = (list((text[0][1:-1]).split(' ')))
        neurons_per_layer= [i[:-1] if i[-1]==',' else i for i in tmp]
        neurons_per_layer = np.array(neurons_per_layer, dtype=np.uint8)
        tmp = (list((text[1][1:-1].replace('\n','')).split(' ')))
        weights = [i for i in tmp if i!='']
        weights = np.array(weights,dtype=np.float)
        return NeuralNetwork(neurons_per_layer, weights)