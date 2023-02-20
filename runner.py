import os
import time

from helpers.paths import *
from codec.simple_decoder import SimpleDecoder
from codec.interpolation_decoder import InterpolationDecoder
from codec.simple_encoder import SimpleEncoder
from codec.interpolation_encoder import InterpolationEncoder
from codec.models import Config, EncodingType, DecodingType
from stats.error_map import ErrorMap
from stats.intensity_map import IntensityMap
from stats.stats import save_json_stats, export_stats_to_excel


def run_codec(sequence_path, config):
    os.makedirs(experiment_dir_path(sequence_path, config.name), exist_ok=True)

    start_time = time.time()

    if config.encoding_type == EncodingType.PICK_INTERPOLATE:
        InterpolationEncoder(sequence_path, config).encode()
    else:
        SimpleEncoder(sequence_path, config).encode()

    mid_time = time.time()

    if config.decoding_type == DecodingType.INTERPOLATE and config.encoding_type in [EncodingType.PICK_REPEAT, EncodingType.PICK_INTERPOLATE]:
        InterpolationDecoder(code_path(sequence_path, config.name), config).decode()
    else:
        SimpleDecoder(code_path(sequence_path, config.name), config).decode()

    end_time = time.time()

    IntensityMap(metadata_path(sequence_path, config.name), config).create()
    ErrorMap(sequence_path, decoded_sequence_path(sequence_path, config.name)).create()

    save_json_stats(sequence_path, config, mid_time - start_time, end_time - mid_time)


def run_bpps_batch():
    sequences_bpps = {
        # '144_176/akiyo.yuv': [1.0, 1.5, 2.0, 2.5],

        # '288_352/akiyo.yuv': [1.0, 1.5, 2.0, 2.5],
        # '288_352/silent.yuv': [1.0, 1.5, 2.0, 2.5],
        # '288_352/soccer.yuv': [1.0, 1.5, 2.0, 2.5],
        # '288_352/bus.yuv': [1.0, 1.5, 2.0, 2.5],
        # '288_352/foreman.yuv': [1.0, 1.5, 2.0, 2.5],
        # '288_352/ice.yuv': [1.0, 1.5, 2.0, 2.5],

        # '1080_1920/crowd_run.yuv': [1.0, 1.5, 2.0, 2.5],
        # '1080_1920/ducks_take_off.yuv': [1.0, 1.5, 2.0, 2.5],
        # '1080_1920/pedestrian_area.yuv': [1.0, 1.5, 2.0, 2.5],
        # '1080_1920/tractor.yuv': [1.0, 1.5, 2.0, 2.5]
    }

    modes = [
        [EncodingType.PICK_REPEAT, DecodingType.REPEAT],
        [EncodingType.PICK_REPEAT, DecodingType.INTERPOLATE],
        [EncodingType.PICK_INTERPOLATE, DecodingType.INTERPOLATE],
        [EncodingType.AVERAGE_REPEAT, DecodingType.REPEAT],
        [EncodingType.AVERAGE_REPEAT, DecodingType.INTERPOLATE],
        [EncodingType.AVERAGE_INTERPOLATE, DecodingType.INTERPOLATE]
    ]

    for sequence, target_bpps in sequences_bpps.items():
        for target_bpp in target_bpps:
            for mode in modes:
                sequence_path = os.path.join(SEQUENCES_DIR, sequence)
                run_codec(sequence_path, Config(16, 16, 16, target_bpp, mode[0], mode[1], ''))

    export_stats_to_excel(RESULTS_DIR)


def run_blocks_batch():
    sequences = [
        # '144_176/akiyo.yuv',
        
        # '288_352/akiyo.yuv',
        # '288_352/silent.yuv',
        # '288_352/soccer.yuv',
        # '288_352/bus.yuv',
        # '288_352/foreman.yuv',
        # '288_352/ice.yuv'

        # '1080_1920/crowd_run.yuv',
        # '1080_1920/ducks_take_off.yuv',
        # '1080_1920/pedestrian_area.yuv',
        # '1080_1920/tractor.yuv'
    ]

    blocks = [
        (8, 8, 4),
        (8, 8, 8),
        (8, 8, 16),
        (16, 16, 4),
        (16, 16, 8),
        (16, 16, 16),
        (32, 32, 1),
        (32, 32, 2),
        (32, 32, 4),
        (64, 64, 1),
        (64, 64, 2),
        (64, 64, 4),
        (128, 128, 1),
        (128, 128, 2),
        (128, 128, 4)
    ]

    modes = [
        [EncodingType.AVERAGE_REPEAT, DecodingType.INTERPOLATE]
    ]

    for sequence in sequences:
        for rows, cols, frames in blocks:
            for mode in modes:
                sequence_path = os.path.join(SEQUENCES_DIR, sequence)
                run_codec(sequence_path, Config(rows, cols, frames, 1.0, mode[0], mode[1], ''))

    export_stats_to_excel(RESULTS_DIR)


if __name__ == '__main__':
    run_bpps_batch()
