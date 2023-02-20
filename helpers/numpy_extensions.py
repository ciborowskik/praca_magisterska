import numpy as np
from scipy.ndimage import zoom

from codec.models import Shape3D


def pick_first_samples(array, chunk: Shape3D):
    return array[::chunk.rows, ::chunk.cols, ::chunk.frames]


def pick_last_samples(array, chunk: Shape3D):
    return array[chunk.rows-1::chunk.rows, chunk.cols-1::chunk.cols, chunk.frames-1::chunk.frames]


def repeat_2d(array, zoom_shape):
    rows, cols = array.shape
    rows_zoom, cols_zoom = zoom_shape

    result = np.empty((rows, rows_zoom, cols, cols_zoom), array.dtype)
    result[...] = array[:, None, :, None]

    return result.reshape(rows * rows_zoom, cols * cols_zoom)


def repeat_3d(array, zoom_shape: Shape3D):
    rows, cols, frames = array.shape
    rows_zoom, cols_zoom, frames_zoom = zoom_shape.rows, zoom_shape.cols, zoom_shape.frames

    result = np.empty((rows, rows_zoom, cols, cols_zoom, frames, frames_zoom), array.dtype)
    result[...] = array[:, None, :, None, :, None]

    return result.reshape(rows * rows_zoom, cols * cols_zoom, frames * frames_zoom)


def zoom_3d(array, target_shape: Shape3D):
    zoom_factors = np.array(target_shape.as_tuple()) / array.shape
    zoomed = zoom(array, zoom_factors, order=1)

    return zoomed


def averages_3d(array, chunk_shape: Shape3D, chunks_count: Shape3D):
    tmp_shape = np.column_stack([chunks_count.as_tuple(), chunk_shape.as_tuple()]).ravel()

    cubes = array.reshape(tmp_shape).transpose(0, 2, 4, 1, 3, 5).reshape(-1, *chunk_shape.as_tuple())
    averages = cubes.mean(axis=(1, 2, 3)).reshape(chunks_count.as_tuple()).astype(np.uint8)

    return averages
