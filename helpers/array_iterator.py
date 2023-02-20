import numpy as np


class ArrayIterator:
    def __init__(self, data):
        self.data = data
        self.position = 0

    def get(self):
        position = self.position
        self.position += 1

        return self.data[position]

    def get_many(self, shape):
        position = self.position
        count = np.prod(shape)
        self.position += count

        return np.reshape(self.data[position:position+count], shape)

    def has_next(self):
        return self.position < self.data.size
