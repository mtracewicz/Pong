import numpy as np

class DataGatherer():
    def __init__(self,data_shape=[5]):
        self._iterator = 0
        self._tmp_iterator = 0
        self._data = np.zeros((tuple([5000]+data_shape)))
        self._data[:] = None
        self._tmp = np.zeros((tuple([500]+data_shape)))
        self._tmp[:] = None

    def record(self, input):
        pass
        # self._tmp[self._tmp_iterator] = input
        # self._tmp_iterator += 1

    def _add_examples(self):
        if self._iterator == self._data.shape[0]:
            self._data.reshape(self._data.shape[0]*2)
        self._data[self._iterator] = input
        self._iterator += 1

    def get_data(self):
        return self._data