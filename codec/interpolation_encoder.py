import numpy as np
from skimage.metrics import mean_squared_error

from helpers.numpy_extensions import pick_first_samples
from helpers.paths import code_path, metadata_path
from codec.interpolation_decoder import InterpolationDecoder
from codec.rd import bisection, convex_hull
from codec.models import Config, Shape3D, SamplingMode, BlockEncodingData
from yuv_io.yuv_reader import YuvReader


class InterpolationEncoder:
    def __init__(self, sequence_path, config: Config):
        self.config = config

        self.source_reader = YuvReader(sequence_path)
        self.source_rows, self.source_cols = self.source_reader.y_shape

        self.code_writer = open(code_path(sequence_path, config.name), 'wb')
        self.metadata_writer = open(metadata_path(sequence_path, config.name), 'wb')

    def encode(self):
        self.metadata_writer.write(self.source_rows.to_bytes(2, 'little'))
        self.metadata_writer.write(self.source_cols.to_bytes(2, 'little'))

        y_part = np.zeros((self.source_rows+1, self.source_cols+1, self.config.block.frames+1), dtype=np.uint8)
        u_part = np.zeros((self.source_rows+1, self.source_cols+1, self.config.block.frames+1), dtype=np.uint8)
        v_part = np.zeros((self.source_rows+1, self.source_cols+1, self.config.block.frames+1), dtype=np.uint8)

        is_first_part = True
        while True:
            # read frames into part
            for i in range(self.config.block.frames):
                frame = self.source_reader.read_next()

                # if there is no more parts - save and return
                if frame is None:
                    self.code_writer.close()
                    self.metadata_writer.close()
                    return

                y_part[1:, 1:, i+1] = frame[:, :, 0]
                u_part[1:, 1:, i+1] = frame[:, :, 1]
                v_part[1:, 1:, i+1] = frame[:, :, 2]

            # duplicate last row and last column on the part edge
            y_part[0, :, :] = y_part[1, :, :]
            u_part[0, :, :] = u_part[1, :, :]
            v_part[0, :, :] = v_part[1, :, :]

            u_part[:, 0, :] = u_part[:, 1, :]
            v_part[:, 0, :] = v_part[:, 1, :]
            y_part[:, 0, :] = y_part[:, 1, :]

            # duplicate first frame on the part edge (only for first part needed)
            if is_first_part:
                y_part[:, :, 0] = y_part[:, :, 1]
                u_part[:, :, 0] = u_part[:, :, 1]
                v_part[:, :, 0] = v_part[:, :, 1]

                is_first_part = False

            # prepare data for each block in part
            params = []

            for r in range(0, self.source_rows, self.config.block.rows):
                for c in range(0, self.source_cols, self.config.block.cols):
                    block = Shape3D(
                        min(self.config.block.rows, self.source_rows - r),
                        min(self.config.block.cols, self.source_cols - c),
                        self.config.block.frames)

                    y_block = y_part[r:r+block.rows+1, c:c+block.cols+1, :]
                    u_block = u_part[r:r+block.rows+1, c:c+block.cols+1, :]
                    v_block = v_part[r:r+block.rows+1, c:c+block.cols+1, :]

                    params.append(BlockEncodingData(y_block, u_block, v_block, block))

            # prepare RD hulls and find best modes
            hulls = [self.get_rd_hull_for_block(args) for args in params]
            mode_ids = bisection(hulls, self.config.target_bpp)

            # encode part with best modes
            for data, mode_id in zip(params, mode_ids):
                mode = self.config.get_mode(mode_id, data.block)

                y_encoded, u_encoded, v_encoded = self.pick_samples(data.y_block, data.u_block, data.v_block, mode)
                code = self.get_code(y_encoded, u_encoded, v_encoded)

                self.code_writer.write(code)
                self.metadata_writer.write(mode_id.to_bytes(1, 'little'))

            # copy last frame from part into first frame in next part
            y_part[:, :, 0] = y_part[:, :, -1]
            u_part[:, :, 0] = u_part[:, :, -1]
            v_part[:, :, 0] = v_part[:, :, -1]

    def get_rd_hull_for_block(self, data: BlockEncodingData):
        modes = self.config.get_modes(data.block)
        rd = np.empty((len(modes), 3))

        for idx, mode in enumerate(modes):
            y_encoded, u_encoded, v_encoded = self.pick_samples(data.y_block, data.u_block, data.v_block, mode)
            y_decoded, u_decoded, v_decoded = InterpolationDecoder.interpolate_samples(y_encoded, u_encoded, v_encoded, mode)

            merged_source = np.hstack((data.y_block[1:, 1:, 1:], data.u_block[1:, 1:, 1:], data.v_block[1:, 1:, 1:]))
            merged_decoded = np.hstack((y_decoded[1:, 1:, 1:], u_decoded[1:, 1:, 1:], v_decoded[1:, 1:, 1:]))

            rd[idx, 0] = idx
            rd[idx, 1] = mode.rate  # R
            rd[idx, 2] = mean_squared_error(merged_source, merged_decoded)  # D

        rd = rd[rd[:, 0] >= 0, :]
        rd = convex_hull(rd)

        return rd

    @staticmethod
    def pick_samples(y_block, u_block, v_block, mode: SamplingMode):
        y_samples = pick_first_samples(y_block, mode.y_chunk)
        u_samples = pick_first_samples(u_block, mode.uv_chunk)
        v_samples = pick_first_samples(v_block, mode.uv_chunk)

        return y_samples, u_samples, v_samples

    @staticmethod
    def get_code(y_encoded, u_encoded, v_encoded):
        y_code = y_encoded[1:, 1:, 1:].flatten()
        u_code = u_encoded[1:, 1:, 1:].flatten()
        v_code = v_encoded[1:, 1:, 1:].flatten()

        return np.hstack((y_code, u_code, v_code))
