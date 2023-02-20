import numpy as np
from skimage.metrics import mean_squared_error

from helpers.numpy_extensions import averages_3d, pick_last_samples
from helpers.paths import code_path, metadata_path
from codec.simple_decoder import SimpleDecoder
from codec.rd import bisection, convex_hull
from codec.models import Config, BlockEncodingData, Shape3D, SamplingMode, EncodingType
from yuv_io.yuv_reader import YuvReader


class SimpleEncoder:
    def __init__(self, sequence_path, config: Config):
        encoding_functions = {
            EncodingType.PICK_REPEAT: self.pick_samples,
            EncodingType.AVERAGE_REPEAT: self.get_average_samples,
            EncodingType.AVERAGE_INTERPOLATE: self.get_average_samples
        }

        decoding_functions = {
            EncodingType.PICK_REPEAT: SimpleDecoder.repeat_samples,
            EncodingType.AVERAGE_REPEAT: SimpleDecoder.repeat_samples,
            EncodingType.AVERAGE_INTERPOLATE: SimpleDecoder.interpolate_average_samples
        }

        self.encoding_function = encoding_functions[config.encoding_type]
        self.decoding_function = decoding_functions[config.encoding_type]
        self.config = config

        self.source_reader = YuvReader(sequence_path)
        self.source_rows, self.source_cols = self.source_reader.y_shape

        self.code_writer = open(code_path(sequence_path, config.name), 'wb')
        self.metadata_writer = open(metadata_path(sequence_path, config.name), 'wb')

    def encode(self):
        self.metadata_writer.write(self.source_rows.to_bytes(2, 'little'))
        self.metadata_writer.write(self.source_cols.to_bytes(2, 'little'))

        while True:
            y_part = np.empty((self.source_rows, self.source_cols, self.config.block.frames), dtype=np.uint8)
            u_part = np.empty((self.source_rows, self.source_cols, self.config.block.frames), dtype=np.uint8)
            v_part = np.empty((self.source_rows, self.source_cols, self.config.block.frames), dtype=np.uint8)

            for i in range(self.config.block.frames):
                frame = self.source_reader.read_next()

                if frame is None:
                    self.code_writer.close()
                    self.metadata_writer.close()
                    return

                y_part[:, :, i] = frame[:, :, 0]
                u_part[:, :, i] = frame[:, :, 1]
                v_part[:, :, i] = frame[:, :, 2]

            params = []

            for r in range(0, self.source_rows, self.config.block.rows):
                for c in range(0, self.source_cols, self.config.block.cols):
                    block = Shape3D(
                        min(self.config.block.rows, self.source_rows - r),
                        min(self.config.block.cols, self.source_cols - c),
                        self.config.block.frames)

                    y_block = y_part[r:r+block.rows, c:c+block.cols, :]
                    u_block = u_part[r:r+block.rows, c:c+block.cols, :]
                    v_block = v_part[r:r+block.rows, c:c+block.cols, :]

                    params.append(BlockEncodingData(y_block, u_block, v_block, block))

            hulls = [self.get_rd_hull_for_block(args) for args in params]
            mode_ids = bisection(hulls, self.config.target_bpp)

            for data, mode_id in zip(params, mode_ids):
                mode = self.config.get_mode(mode_id, data.block)

                y_encoded, u_encoded, v_encoded = self.encoding_function(data.y_block, data.u_block, data.v_block, mode)
                code = self.get_code(y_encoded, u_encoded, v_encoded)

                self.code_writer.write(code)
                self.metadata_writer.write(mode_id.to_bytes(1, 'little'))

    def get_rd_hull_for_block(self, data: BlockEncodingData):
        modes = self.config.get_modes(data.block)
        rd = np.empty((len(modes), 3))

        for idx, mode in enumerate(modes):
            y_encoded, u_encoded, v_encoded = self.encoding_function(data.y_block, data.u_block, data.v_block, mode)
            y_decoded, u_decoded, v_decoded = self.decoding_function(y_encoded, u_encoded, v_encoded, mode)

            merged_source = np.hstack((data.y_block, data.u_block, data.v_block))
            merged_decoded = np.hstack((y_decoded, u_decoded, v_decoded))

            rd[idx, 0] = mode.idx
            rd[idx, 1] = mode.rate  # R
            rd[idx, 2] = mean_squared_error(merged_source, merged_decoded)  # D

        rd = convex_hull(rd)

        return rd

    @staticmethod
    def pick_samples(y_block, u_block, v_block, mode: SamplingMode):
        y_samples = pick_last_samples(y_block, mode.y_chunk)
        u_samples = pick_last_samples(u_block, mode.uv_chunk)
        v_samples = pick_last_samples(v_block, mode.uv_chunk)

        return y_samples, u_samples, v_samples

    @staticmethod
    def get_average_samples(y_block, u_block, v_block, mode: SamplingMode):
        y_samples = averages_3d(y_block, mode.y_chunk, mode.y_points)
        u_samples = averages_3d(u_block, mode.uv_chunk, mode.uv_points)
        v_samples = averages_3d(v_block, mode.uv_chunk, mode.uv_points)

        return y_samples, u_samples, v_samples

    @staticmethod
    def get_code(y_encoded, u_encoded, v_encoded):
        y_code = y_encoded.flatten()
        u_code = u_encoded.flatten()
        v_code = v_encoded.flatten()

        return np.hstack((y_code, u_code, v_code))
