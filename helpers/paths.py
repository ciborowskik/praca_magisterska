import os.path
from os import path

import numpy as np

SEQUENCES_DIR = path.normpath(path.join(__file__, '../../_sequences'))
RESULTS_DIR = path.normpath(path.join(__file__, '../../_results'))

SAMPLE_SEQUENCE_PATH = path.normpath(path.join(SEQUENCES_DIR, '144_176/akiyo.yuv'))


def modify_path(file_path, experiment_name, extension):
    if experiment_name:
        file_path = path.join(experiment_dir_path(file_path, experiment_name), path.basename(file_path))

    return f'{path.splitext(file_path)[0]}.{extension}'


def experiment_dir_path(sequence_path, experiment_name):
    return path.join(RESULTS_DIR, path.basename(path.dirname(sequence_path)), experiment_name)


def code_path(file_path, experiment_name=None):
    return modify_path(file_path, experiment_name, 'code')


def metadata_path(file_path, experiment_name=None):
    return modify_path(file_path, experiment_name, 'meta')


def decoded_sequence_path(file_path, experiment_name=None):
    return modify_path(file_path, experiment_name, 'yuv_decoded')


def intensity_map_path(file_path, experiment_name=None):
    return modify_path(file_path, experiment_name, 'intensity_map')


def error_map_path(file_path, experiment_name=None):
    return modify_path(file_path, experiment_name, 'error_map')


def stats_path(file_path, experiment_name=None):
    return modify_path(file_path, experiment_name, 'stats')


def video_shape(file_path):
    shape_dir = path.dirname(file_path) if file_path.endswith('.yuv') else path.dirname(path.dirname(file_path))
    shape_data = path.basename(shape_dir).split('_')

    return int(shape_data[0]), int(shape_data[1])


def frames_count(sequence_path):
    sequence_size = os.path.getsize(sequence_path) * 2
    frame_size = np.prod(video_shape(sequence_path)) * 3

    return int(sequence_size / frame_size)
