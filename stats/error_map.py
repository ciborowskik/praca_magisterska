import numpy as np

from helpers.paths import error_map_path
from yuv_io.maps_writer import MapsWriter
from yuv_io.yuv_reader import YuvReader


class ErrorMap:
    def __init__(self, sequence_path, decoded_sequence_path):
        self.sequence_reader = YuvReader(sequence_path)
        self.decoded_reader = YuvReader(decoded_sequence_path)
        self.writer = MapsWriter(error_map_path(decoded_sequence_path))

    def create(self):
        while True:
            frame = self.sequence_reader.read_next()
            decoded = self.decoded_reader.read_next()

            if frame is None or decoded is None:
                break

            diff = np.abs(frame.astype(int) - decoded.astype(int))
            diff = np.sum(diff, axis=2) * 10
            diff = np.clip(diff, 0, 255).astype(np.uint8)

            self.writer.write_next(diff)

        self.writer.close()
