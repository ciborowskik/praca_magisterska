from enum import Enum
from math import log2


class EncodingType(Enum):
    PICK_REPEAT = 1
    PICK_INTERPOLATE = 2
    AVERAGE_REPEAT = 3
    AVERAGE_INTERPOLATE = 4


class DecodingType(Enum):
    REPEAT = 1
    INTERPOLATE = 2


class Shape3D:
    def __init__(self, rows, cols, frames):
        self.rows = int(rows)
        self.cols = int(cols)
        self.frames = int(frames)
        self.tuple = (self.rows, self.cols, self.frames)
        self.count = self.rows * self.cols * self.frames

    def as_tuple(self):
        return self.tuple

    def is_divisible(self, other):
        return self.rows % other.rows == 0 and self.cols % other.cols == 0 and self.frames % other.frames == 0

    def __str__(self):
        return f'({self.rows}, {self.cols}, {self.frames})'

    def __eq__(self, other):
        return self.rows == other.rows and self.cols == other.cols and self.frames == other.frames

    def __add__(self, n):
        return Shape3D(self.rows + n, self.cols + n, self.frames + n)

    def __floordiv__(self, other):
        return Shape3D(self.rows // other.rows, self.cols // other.cols, self.frames // other.frames)


class SamplingMode:
    def __init__(self, idx: int, y_chunk: Shape3D, uv_chunk: Shape3D, block: Shape3D):
        self.idx = idx

        self.y_chunk: Shape3D = y_chunk
        self.y_points: Shape3D = block // y_chunk

        self.uv_chunk: Shape3D = uv_chunk
        self.uv_points: Shape3D = block // uv_chunk

        self.block: Shape3D = block

        self.rate = (self.y_points.count + self.uv_points.count * 2) / (self.block.count * 3) * 24


class Config:
    def __init__(self, rows: int, cols: int, frames: int, target_bpp: float, encoding: EncodingType, decoding: DecodingType, info: str):
        name = f'{rows}__{cols}__{frames}__{target_bpp}__{encoding.name.lower()}__{decoding.name.lower()}  {info}'.strip()

        self.name: str = name
        self.block: Shape3D = Shape3D(rows, cols, frames)
        self.target_bpp: float = target_bpp
        self.encoding_type: EncodingType = encoding
        self.decoding_type: DecodingType = decoding
        self.chunks = self.generate_chunks(self.block)
        self.modes = self.generate_modes(self.chunks, self.block)

    def get_mode(self, idx, block: Shape3D):
        if block == self.block:
            return self.modes[idx]

        y_chunk, uv_chunk = self.chunks[idx]

        return SamplingMode(idx, y_chunk, uv_chunk, block)

    def get_modes(self, block: Shape3D):
        if block == self.block:
            return self.modes

        return self.generate_modes(self.chunks, block)

    @staticmethod
    def generate_modes(chunks, block):
        modes = []

        for idx, chunk in enumerate(chunks):
            y_chunk, uv_chunk = chunk

            if block.is_divisible(uv_chunk):
                modes.append(SamplingMode(idx, y_chunk, uv_chunk, block))

        return modes

    @staticmethod
    def generate_chunks(block):
        intensities = []

        for r in range(int(log2(block.rows))):
            for c in range(int(log2(block.cols))):
                if block.frames == 1:
                    intensities.append((Shape3D(2**r, 2**c, 1), Shape3D(2**r*2, 2**c*2, 1)))
                else:
                    for f in range(int(log2(block.frames))):
                        intensities.append((Shape3D(2**r, 2**c, 2**f), Shape3D(2**r*2, 2**c*2, 2**f*2)))

        return intensities


class BlockEncodingData:
    def __init__(self, y_block, u_block, v_block, block: Shape3D):
        self.y_block = y_block
        self.u_block = u_block
        self.v_block = v_block
        self.block = block
