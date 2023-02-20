import numpy as np

from helpers.array_iterator import ArrayIterator
from helpers.paths import video_shape


class MapsReader:
    def __init__(self, sequence_path, bgr_channel):
        with open(sequence_path, 'rb') as file:
            self.data = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        self.shape = video_shape(sequence_path)
        self.frame_bytes = np.prod(self.shape)
        self.bgr_channel = bgr_channel

    def read_next(self):
        if not self.data.has_next():
            return None

        bgr_frame = np.zeros(self.shape + (3,), dtype=np.uint8)
        bgr_frame[:, :, self.bgr_channel] = self.data.get_many(self.shape)

        return bgr_frame
