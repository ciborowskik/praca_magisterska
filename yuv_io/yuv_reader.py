import cv2 as cv
import numpy as np

from helpers.array_iterator import ArrayIterator
from helpers.numpy_extensions import repeat_2d
from helpers.paths import video_shape


class YuvReader:
    def __init__(self, sequence_path):
        with open(sequence_path, 'rb') as file:
            self.data = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        self.y_shape = video_shape(sequence_path)
        self.uv_shape = (self.y_shape[0] // 2, self.y_shape[1] // 2)

    def read_next(self):
        if not self.data.has_next():
            return None

        y = self.data.get_many(self.y_shape)
        u = repeat_2d(self.data.get_many(self.uv_shape), (2, 2))
        v = repeat_2d(self.data.get_many(self.uv_shape), (2, 2))

        return np.dstack((y, u, v))

    def read_next_as_bgr(self):
        frame = self.read_next()

        return cv.cvtColor(frame, cv.COLOR_YUV2BGR) if frame is not None else frame
