from random import randint
import numpy as np

class NeuralNetwork():
    def __init__(self, neurons_per_layer = np.array([6, 5, 4]), weights = None):
        self._neurons_per_layer = np.concatenate((neurons_per_layer,np.array([1])))
        number_of_weights = int(np.sum(np.array([self._neurons_per_layer[i]*self._neurons_per_layer[i+1] for i in range(len(neurons_per_layer))])))
        self._weights = weights if weights is not None else (np.random.rand(number_of_weights) - 0.5)/100

    def fit(self, inputs, outputs, learning_rate = 0.05):
        pass

    def predict(self, inputs):
        value_on_layer = np.array(inputs)
        current = 0
        tmp=[]
        prev = 0
        for t,layer in enumerate(self._neurons_per_layer[1:]):
            for i in range(layer):
                tmp.append(np.sum(value_on_layer*self._weights[current+i*layer+i:current+(i+1)*layer+i+1 if layer != 1 else current+prev]))
            current += len(value_on_layer)*layer-t
            value_on_layer = tmp.copy()
            prev = layer
            tmp = []
        a = value_on_layer[0]
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