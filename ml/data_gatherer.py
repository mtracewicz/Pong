import numpy as np


class DataGatherer():
    def __init__(self):
        self._data = []
        self._tmp = []

    def record(self, input):
        self._tmp.append(input)

    def save(self):
        self._data.append(self._tmp.copy())

    def discard(self):
        self._tmp = []

    def get_data(self):
        return np.array(self._data)
