import os
from datetime import datetime
from os import path
from time import time

import cv2 as cv
import numpy as np

from codec.models import Shape3D
from helpers.numpy_extensions import repeat_3d
from helpers.paths import decoded_sequence_path, intensity_map_path, error_map_path
from yuv_io.maps_reader import MapsReader
from yuv_io.yuv_reader import YuvReader


class YuvPlayer:
    def __init__(self, sequence_path, config, play_original, play_decoded, play_samples, play_error, zoom, fps):
        self.sequence_name = path.basename(sequence_path)
        self.play_original = play_original
        self.play_decoded = play_decoded
        self.play_samples = play_samples
        self.play_error = play_error
        self.block = config.block
        self.zoom = zoom
        self.fps = fps

        self.original_reader = YuvReader(sequence_path) if play_original else None
        self.decoded_reader = YuvReader(decoded_sequence_path(sequence_path, config.name)) if play_decoded else None
        self.samples_reader = MapsReader(intensity_map_path(sequence_path, config.name), 1) if play_samples else None
        self.error_reader = MapsReader(error_map_path(sequence_path, config.name), 2) if play_error else None

    def play(self):
        current_display = None

        while True:
            start_time = time()

            to_display = []

            if self.play_original:
                to_display.append(self.original_reader.read_next_as_bgr())
            if self.play_samples:
                to_display.append(self.samples_reader.read_next())
            if self.play_error:
                to_display.append(self.error_reader.read_next())
            if self.play_decoded:
                to_display.append(self.decoded_reader.read_next_as_bgr())

            if any(frame is None for frame in to_display):
                break

            current_display = np.hstack(to_display)

            if self.zoom > 1:
                current_display = repeat_3d(current_display, Shape3D(self.zoom, self.zoom, 1))
                current_display[self.block.rows*self.zoom::self.block.rows*self.zoom, :, :] = 255
                current_display[:, self.block.cols*self.zoom::self.block.cols*self.zoom, :] = 255

            cv.imshow(self.sequence_name, current_display)

            execution_period = (time() - start_time) * 1000
            sleep_period = max(int(1000 / self.fps - execution_period), 1)

            key = cv.waitKey(sleep_period)
            # return if window closed with CROSS
            if not cv.getWindowProperty(self.sequence_name, cv.WND_PROP_VISIBLE):
                return
            # save screenshot if ENTER clicked
            if key == 13:
                self.save_screenshot(current_display)
            # pause if SPACE clicked
            if key == 32:
                while True:
                    key = cv.waitKey()
                    # save screenshot if ENTER clicked
                    if key == 13:
                        self.save_screenshot(current_display)
                    # play if SPACE clicked
                    if key == 32:
                        break

        while True:
            key = cv.waitKey()
            # return if window closed with CROSS
            if not cv.getWindowProperty(self.sequence_name, cv.WND_PROP_VISIBLE):
                return
            # save screenshot if ENTER clicked
            if key == 13:
                self.save_screenshot(current_display)

    @staticmethod
    def save_screenshot(current_display):
        os.makedirs('screenshots', exist_ok=True)
        cv.imwrite(f'screenshots/{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', current_display)
