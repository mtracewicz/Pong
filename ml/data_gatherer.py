import os

import numpy as np


class DataGatherer():
    def __init__(self, data=None):
        self._data = data if data is not None else np.zeros((1, 5))
        self._tmp = []
        self.learn = False

    def record(self, input):
        self._tmp.append(input)

    def save(self):
        tmp = np.array(self._tmp.copy())
        tmp = tmp[::10]
        tmp[:, -1] = tmp[-1, -1]
        if tmp[-1, 0] < 0.5:
            tmp[:, 0] = 1-tmp[:, 0]

        tmp[:, 2] = abs(tmp[:, 2])
        self._data = np.append(self._data, tmp, axis=0)
        self.discard()
        self.learn = True

    def discard(self):
        self._tmp = []

    def get_data(self):
        return np.array(self._data)[1:]

    def save_data(self):
        np.save(os.path.join("model", "data.npy"), self._data)
