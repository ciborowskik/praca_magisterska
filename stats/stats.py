import glob
import json
import os
from datetime import datetime

import pandas as pd
from janitor import xlsx_table
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from helpers.paths import *
from codec.models import Config
from yuv_io.yuv_reader import YuvReader


EXCEL_SHEET_NAME = 'CodecStats'
EXCEL_TABLE_NAME = 'CodecStatsTable'

PROP_SEQUENCE_PATH = 'Sequence path'
PROP_EXPERIMENT_NAME = 'Experiment name'
PROP_ENCODING_MODE = 'Encoding mode'
PROP_DECODING_MODE = 'Decoding mode'
PROP_BLOCK_SIZE = 'Block shape'
PROP_TARGET_BPP = 'Target BPP'
PROP_BITS_PER_PIXEL = 'BPP'
PROP_BITS_PER_PIXEL_INCLUDING_META = 'BPP (including meta)'
PROP_COMPRESSION_RATE_INCLUDING_META = 'CR (including meta)'
PROP_MSE = 'MSE'
PROP_PSNR = 'PSNR [dB]'
PROP_SSIM = 'SSIM'
PROP_RESOLUTION = 'Resolution'
PROP_FRAMES_COUNT = 'Frames count'
PROP_SEQUENCE_SIZE = 'Sequence size [bytes]'
PROP_CODE_SIZE = 'Code size [bytes]'
PROP_METADATA_SIZE = 'Metadata size [bytes]'
PROP_ENCODING_TIME = 'Encoding time [s]'
PROP_DECODING_TIME = 'Decoding time [s]'


def create_excel(df, stats_file_path):
    writer = pd.ExcelWriter(stats_file_path, engine='xlsxwriter')

    add_table_to_excel(writer, df)

    writer.close()


def add_table_to_excel(writer, dataframe):
    dataframe.to_excel(writer, sheet_name=EXCEL_SHEET_NAME, startrow=1, header=False, index=False)

    column_settings = []
    for header in dataframe.columns:
        column_settings.append({'header': header})

    (max_row, max_col) = dataframe.shape
    worksheet = writer.sheets[EXCEL_SHEET_NAME]
    worksheet.add_table(0, 0, max_row, max_col-1, {'name': EXCEL_TABLE_NAME, 'columns': column_settings})
    worksheet.set_column(0, 1, 25)
    worksheet.set_column(1, 2, 75)
    worksheet.set_column(2, max_col, 20)
    worksheet.freeze_panes(0, 2)


def calculate_metrics(sequence_path, experiment_name):
    sequence_reader = YuvReader(sequence_path)
    decoded_sequence_reader = YuvReader(decoded_sequence_path(sequence_path, experiment_name))

    frames_number = frames_count(decoded_sequence_path(sequence_path, experiment_name))

    mse_for_frames = np.zeros(frames_number)
    ssim_for_frames = np.zeros(frames_number)
    psnr_for_frames = np.zeros(frames_number)

    for i in range(frames_number):
        original_frame = sequence_reader.read_next()
        decoded_frame = decoded_sequence_reader.read_next()

        mse_for_frames[i] = mean_squared_error(original_frame, decoded_frame)
        psnr_for_frames[i] = peak_signal_noise_ratio(original_frame, decoded_frame)
        ssim_for_frames[i] = structural_similarity(original_frame, decoded_frame, channel_axis=2)

    mse = mse_for_frames.mean()
    psnr = psnr_for_frames.mean()
    ssim = ssim_for_frames.mean()

    return mse, psnr, ssim


def save_json_stats(sequence_path, config: Config, coding_time, decoding_time):
    sequence_size = 2 * os.path.getsize(sequence_path)
    code_size = os.path.getsize(code_path(sequence_path, config.name))
    metadata_size = os.path.getsize(metadata_path(sequence_path, config.name))
    resolution = video_shape(sequence_path)

    mse, psnr, ssim = calculate_metrics(sequence_path, config.name)

    stats = {
        PROP_SEQUENCE_PATH: os.path.relpath(sequence_path, SEQUENCES_DIR),
        PROP_EXPERIMENT_NAME: config.name,
        PROP_ENCODING_MODE: config.encoding_type.name.lower(),
        PROP_DECODING_MODE: config.decoding_type.name.lower(),
        PROP_BLOCK_SIZE: f'[{config.block.rows}, {config.block.cols}, {config.block.frames}]',
        PROP_TARGET_BPP: config.target_bpp,
        PROP_BITS_PER_PIXEL: code_size / sequence_size * 24,
        PROP_BITS_PER_PIXEL_INCLUDING_META: (code_size + metadata_size) / sequence_size * 24,
        PROP_COMPRESSION_RATE_INCLUDING_META: sequence_size / (code_size + metadata_size),
        PROP_MSE: mse,
        PROP_PSNR: psnr,
        PROP_SSIM: ssim,
        PROP_RESOLUTION: f'{resolution[0]}x{resolution[1]}',
        PROP_FRAMES_COUNT: frames_count(sequence_path),
        PROP_SEQUENCE_SIZE: sequence_size,
        PROP_CODE_SIZE: code_size,
        PROP_METADATA_SIZE: metadata_size,
        PROP_ENCODING_TIME: round(coding_time, 2),
        PROP_DECODING_TIME: round(decoding_time, 2)
    }

    json.dump(stats, open(stats_path(sequence_path, config.name), 'w'), indent=4)


def export_stats_to_excel(results_dir):
    stats_paths = []

    for path_, _, files in os.walk(results_dir):
        stats_paths.extend([os.path.join(path_, file) for file in files if file.endswith('.stats')])

    stats_list = [json.load(open(file_path)) for file_path in stats_paths]

    df = pd.DataFrame(stats_list)
    stats_file_path = os.path.join(RESULTS_DIR, f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

    create_excel(df, stats_file_path)


def import_last_excel_into_df():
    list_of_files = glob.glob(os.path.join(RESULTS_DIR, 'stats_*.xlsx'))
    latest_file = max(list_of_files, key=os.path.getctime)
    df = xlsx_table(latest_file, sheetname=EXCEL_SHEET_NAME, table=EXCEL_TABLE_NAME)

    return df
