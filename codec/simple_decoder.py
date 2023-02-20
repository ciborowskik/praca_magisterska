import numpy as np
from scipy.ndimage import map_coordinates

from helpers.array_iterator import ArrayIterator
from helpers.numpy_extensions import repeat_3d
from helpers.paths import metadata_path, decoded_sequence_path
from codec.models import Config, Shape3D, SamplingMode, DecodingType
from yuv_io.yuv_writer import YuvWriter


class SimpleDecoder:
    def __init__(self, code_path, config: Config):
        decoding_functions = {
            DecodingType.REPEAT: SimpleDecoder.repeat_samples,
            DecodingType.INTERPOLATE: SimpleDecoder.interpolate_average_samples
        }

        self.config = config
        self.decoding_function = decoding_functions[config.decoding_type]
        
        with open(code_path, 'rb') as file:
            self.code = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        with open(metadata_path(code_path), 'rb') as file:
            self.source_rows = np.fromfile(file, dtype=np.uint16, count=1)[0]
            self.source_cols = np.fromfile(file, dtype=np.uint16, count=1)[0]
            self.metadata = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        self.writer = YuvWriter(decoded_sequence_path(code_path))

    def decode(self):
        while self.code.has_next():
            y_part = np.empty((self.source_rows, self.source_cols, self.config.block.frames), dtype=np.uint8)
            u_part = np.empty((self.source_rows, self.source_cols, self.config.block.frames), dtype=np.uint8)
            v_part = np.empty((self.source_rows, self.source_cols, self.config.block.frames), dtype=np.uint8)

            for r in range(0, self.source_rows, self.config.block.rows):
                for c in range(0, self.source_cols, self.config.block.cols):
                    block = Shape3D(
                        min(self.config.block.rows, self.source_rows - r),
                        min(self.config.block.cols, self.source_cols - c),
                        self.config.block.frames)

                    mode_id = self.metadata.get()
                    mode = self.config.get_mode(mode_id, block)

                    y_code = self.code.get_many(mode.y_points.as_tuple())
                    u_code = self.code.get_many(mode.uv_points.as_tuple())
                    v_code = self.code.get_many(mode.uv_points.as_tuple())

                    y_block, u_block, v_block = self.decoding_function(y_code, u_code, v_code, mode)

                    y_part[r:r+block.rows, c:c+block.cols, :] = y_block
                    u_part[r:r+block.rows, c:c+block.cols, :] = u_block
                    v_part[r:r+block.rows, c:c+block.cols, :] = v_block

            for i in range(self.config.block.frames):
                frame = np.dstack((y_part[:, :, i], u_part[:, :, i], v_part[:, :, i]))
                self.writer.write_next(frame)

        self.writer.close()

    @staticmethod
    def repeat_samples(y_input, u_input, v_input, mode: SamplingMode):
        y_block = repeat_3d(y_input, mode.y_chunk)
        u_block = repeat_3d(u_input, mode.uv_chunk)
        v_block = repeat_3d(v_input, mode.uv_chunk)

        return y_block, u_block, v_block

    @staticmethod
    def interpolate_average_samples(y_input, u_input, v_input, mode: SamplingMode):
        y_rows_coord = np.linspace(0.5, mode.block.rows-0.5, mode.block.rows) / mode.y_chunk.rows - 0.5
        y_cols_coord = np.linspace(0.5, mode.block.cols-0.5, mode.block.cols) / mode.y_chunk.cols - 0.5
        y_frames_coord = np.linspace(0.5, mode.block.frames-0.5, mode.block.frames) / mode.y_chunk.frames - 0.5

        y_coord = np.stack(np.meshgrid(y_rows_coord, y_cols_coord, y_frames_coord, indexing='ij'))

        y_block = map_coordinates(y_input, y_coord, order=1, mode='nearest')
        u_block = repeat_3d(u_input, mode.uv_chunk)
        v_block = repeat_3d(v_input, mode.uv_chunk)

        return y_block, u_block, v_block
