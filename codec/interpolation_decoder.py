import numpy as np

from helpers.array_iterator import ArrayIterator
from helpers.numpy_extensions import zoom_3d, pick_first_samples
from helpers.paths import metadata_path, decoded_sequence_path
from codec.models import Config, Shape3D, SamplingMode
from yuv_io.yuv_writer import YuvWriter


class InterpolationDecoder:
    def __init__(self, code_path, config: Config):
        self.config = config

        with open(code_path, 'rb') as file:
            self.code = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        with open(metadata_path(code_path), 'rb') as file:
            self.source_rows = np.fromfile(file, dtype=np.uint16, count=1)[0]
            self.source_cols = np.fromfile(file, dtype=np.uint16, count=1)[0]
            self.metadata = ArrayIterator(np.fromfile(file, dtype=np.uint8))

        self.writer = YuvWriter(decoded_sequence_path(code_path))

    def decode(self):
        y_part = np.empty((self.source_rows+1, self.source_cols+1, self.config.block.frames+1), dtype=np.uint8)
        u_part = np.empty((self.source_rows+1, self.source_cols+1, self.config.block.frames+1), dtype=np.uint8)
        v_part = np.empty((self.source_rows+1, self.source_cols+1, self.config.block.frames+1), dtype=np.uint8)

        is_first_part = True
        while self.code.has_next():
            for r in range(0, self.source_rows, self.config.block.rows):
                for c in range(0, self.source_cols, self.config.block.cols):
                    block = Shape3D(
                        min(self.config.block.rows, self.source_rows - r),
                        min(self.config.block.cols, self.source_cols - c),
                        self.config.block.frames)

                    mode_id = self.metadata.get()
                    mode = self.config.get_mode(mode_id, block)

                    # get extended input
                    y_input = pick_first_samples(y_part[r:r+block.rows+1, c:c+block.cols+1, :], mode.y_chunk)
                    u_input = pick_first_samples(u_part[r:r+block.rows+1, c:c+block.cols+1, :], mode.uv_chunk)
                    v_input = pick_first_samples(v_part[r:r+block.rows+1, c:c+block.cols+1, :], mode.uv_chunk)

                    # fit code into extended input
                    y_input[1:, 1:, 1:] = self.code.get_many(mode.y_points.as_tuple())
                    u_input[1:, 1:, 1:] = self.code.get_many(mode.uv_points.as_tuple())
                    v_input[1:, 1:, 1:] = self.code.get_many(mode.uv_points.as_tuple())

                    # duplicate code values in edge cases
                    if is_first_part:
                        y_input[:, :, 0] = y_input[:, :, 1]
                        u_input[:, :, 0] = u_input[:, :, 1]
                        v_input[:, :, 0] = v_input[:, :, 1]

                    if r == 0:
                        y_input[0, :, :] = y_input[1, :, :]
                        u_input[0, :, :] = u_input[1, :, :]
                        v_input[0, :, :] = v_input[1, :, :]

                    if c == 0:
                        y_input[:, 0, :] = y_input[:, 1, :]
                        u_input[:, 0, :] = u_input[:, 1, :]
                        v_input[:, 0, :] = v_input[:, 1, :]

                    # decode and fill part with proper data
                    y_block, u_block, v_block = self.interpolate_samples(y_input, u_input, v_input, mode)

                    y_part[r+1:r+block.rows+1, c+1:c+block.cols+1, 1:] = y_block[1:, 1:, 1:]
                    u_part[r+1:r+block.rows+1, c+1:c+block.cols+1, 1:] = u_block[1:, 1:, 1:]
                    v_part[r+1:r+block.rows+1, c+1:c+block.cols+1, 1:] = v_block[1:, 1:, 1:]

            # save ready part
            for i in range(self.config.block.frames):
                frame = np.dstack((y_part[1:, 1:, i+1], u_part[1:, 1:, i+1], v_part[1:, 1:, i+1]))
                self.writer.write_next(frame)

            # copy last frame from part into first frame in next part
            y_part[:, :, 0] = y_part[:, :, -1]
            u_part[:, :, 0] = u_part[:, :, -1]
            v_part[:, :, 0] = v_part[:, :, -1]

            is_first_part = False

        self.writer.close()

    @staticmethod
    def interpolate_samples(y_input, u_input, v_input, mode: SamplingMode):
        y_block = zoom_3d(y_input, mode.block + 1)
        u_block = zoom_3d(u_input, mode.block + 1)
        v_block = zoom_3d(v_input, mode.block + 1)

        return y_block, u_block, v_block

