import numpy as np


class DataGatherer():
    def __init__(self):
        self._data = np.zeros((1, 5))
        self._tmp = []
        self.learn = False

    def record(self, input):
        self._tmp.append(input)

    def save(self):
        tmp = np.array(self._tmp.copy())
        tmp[:, -1] = tmp[-1, -1]
        self._data = np.append(self._data, tmp, axis=0)
        self.learn = True

    def discard(self):
        self._tmp = []

    def get_data(self):
        return np.array(self._data)
