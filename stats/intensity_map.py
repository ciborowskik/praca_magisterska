import numpy as np

from helpers.array_iterator import ArrayIterator
from helpers.paths import intensity_map_path
from codec.models import Config, Shape3D
from yuv_io.maps_writer import MapsWriter


class IntensityMap:
    def __init__(self, metadata_path, config: Config):
        self.block: Shape3D = config.block
        self.config = config
        
        with open(metadata_path, 'rb') as file:
            self.source_rows = np.fromfile(file, dtype=np.uint16, count=1)[0]
            self.source_cols = np.fromfile(file, dtype=np.uint16, count=1)[0]
            self.metadata = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        self.writer = MapsWriter(intensity_map_path(metadata_path))

    def create(self):
        while self.metadata.has_next():
            y_part = np.zeros((self.source_rows, self.source_cols, self.block.frames), dtype=np.uint8)

            for r in range(0, self.source_rows, self.block.rows):
                for c in range(0, self.source_cols, self.block.cols):
                    block = Shape3D(
                        min(self.block.rows, self.source_rows - r),
                        min(self.block.cols, self.source_cols - c),
                        self.block.frames)

                    mode_id = self.metadata.get()
                    mode = self.config.get_mode(mode_id, block)

                    y_part[
                        r+mode.y_chunk.rows-1:r+block.rows:mode.y_chunk.rows,
                        c+mode.y_chunk.cols-1:c+block.cols:mode.y_chunk.cols,
                        mode.y_chunk.frames-1::mode.y_chunk.frames] = 255

            for i in range(self.block.frames):
                self.writer.write_next(y_part[:, :, i])

        self.writer.close()
