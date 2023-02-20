import os
import sys
from os import path

from PyQt6.QtCore import Qt, QRunnable, pyqtSlot, QThreadPool
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import *

from helpers.paths import SEQUENCES_DIR, SAMPLE_SEQUENCE_PATH, decoded_sequence_path, intensity_map_path, \
    RESULTS_DIR, stats_path, error_map_path
from codec.models import EncodingType, Config, DecodingType
from runner import run_codec, export_stats_to_excel
from yuv_io.yuv_player import YuvPlayer


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Mgr')
        self.setGeometry(100, 100, 0, 0)

        self.fps = 30
        self.zoom = 1
        self.sequence_path = SAMPLE_SEQUENCE_PATH
        self.config: Config = None
        self.threadpool = QThreadPool()

        parameters_section = QGroupBox('Parameters')
        single_test_section = QGroupBox('Test and watch single sequence')
        stats_section = QGroupBox('Stats')

        parameters_section.setLayout(self.prepare_parameters_section())
        single_test_section.setLayout(self.prepare_single_test_section())
        stats_section.setLayout(self.prepare_stats_section())

        main_layout = QVBoxLayout()
        main_layout.setSpacing(40)
        main_layout.addWidget(parameters_section)
        main_layout.addWidget(single_test_section)
        main_layout.addWidget(stats_section)
        main_layout.addWidget(QWidget())

        self.main_widget = QWidget()
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.main_widget)

        self.update_config_and_buttons_availability()

    def prepare_parameters_section(self):
        row = 0
        layout = QGridLayout()
        layout.setColumnMinimumWidth(0, 200)
        layout.setColumnMinimumWidth(1, 200)

        tmp = QLabel('Target BPP')
        layout.addWidget(tmp, row, 0)

        self.target_bpp_box = QLineEdit(self)
        self.target_bpp_box.setValidator(QDoubleValidator(0.00, 5.00, 2))
        self.target_bpp_box.setText('1')
        self.target_bpp_box.textChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.target_bpp_box, row, 1)
        row += 1

        tmp = QLabel('Block rows')
        layout.addWidget(tmp, row, 0)

        self.rows_box = QComboBox(self)
        self.rows_box.addItems(['8', '16', '32', '64', '128'])
        self.rows_box.setCurrentIndex(1)
        self.rows_box.currentTextChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.rows_box, row, 1)
        row += 1

        tmp = QLabel('Block cols')
        layout.addWidget(tmp, row, 0)

        self.cols_box = QComboBox(self)
        self.cols_box.addItems(['8', '16', '32', '64', '128'])
        self.cols_box.setCurrentIndex(1)
        self.cols_box.currentTextChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.cols_box, row, 1)
        row += 1

        tmp = QLabel('Block frames')
        layout.addWidget(tmp, row, 0)

        self.frames_box = QComboBox(self)
        self.frames_box.addItems(['1', '2', '4', '8', '16'])
        self.frames_box.setCurrentIndex(4)
        self.frames_box.currentTextChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.frames_box, row, 1)
        row += 1

        tmp = QLabel('Encoding mode')
        layout.addWidget(tmp, row, 0)

        self.encoding_box = QComboBox(self)
        for item in EncodingType:
            self.encoding_box.addItem(item.name.lower(), item)
        self.encoding_box.setCurrentIndex(2)
        self.encoding_box.currentTextChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.encoding_box, row, 1)
        row += 1

        tmp = QLabel('Decoding mode')
        layout.addWidget(tmp, row, 0)

        self.decoding_box = QComboBox(self)
        for item in DecodingType:
            self.decoding_box.addItem(item.name.lower(), item)
        self.decoding_box.setCurrentIndex(1)
        self.decoding_box.currentTextChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.decoding_box, row, 1)
        row += 1

        tmp = QLabel('Additional info')
        layout.addWidget(tmp, row, 0)

        self.additional_info_box = QLineEdit(self)
        self.additional_info_box.textChanged.connect(self.update_config_and_buttons_availability)
        layout.addWidget(self.additional_info_box, row, 1)

        return layout

    def prepare_single_test_section(self):
        row = 0
        layout = QGridLayout()
        layout.setColumnMinimumWidth(0, 200)
        layout.setColumnMinimumWidth(1, 200)

        tmp = QPushButton('Choose sequence')
        tmp.clicked.connect(self.show_file_dialog_handler)
        layout.addWidget(tmp, row, 0)

        self.sequence_path_label = QLineEdit(path.relpath(self.sequence_path, start=SEQUENCES_DIR))
        self.sequence_path_label.setDisabled(True)
        self.sequence_path_label.setFixedHeight(20)
        layout.addWidget(self.sequence_path_label, row, 1)
        row += 1

        layout.setRowMinimumHeight(row, 10)
        row += 1

        tmp = QPushButton('Run codec')
        tmp.clicked.connect(self.run_codec_handler)
        layout.addWidget(tmp, row, 0, 1, 2)
        row += 1

        layout.setRowMinimumHeight(row, 20)
        row += 1

        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setMinimum(1)
        slider.setMaximum(30)
        slider.setValue(self.fps)
        slider.valueChanged.connect(self.set_fps_handler)
        layout.addWidget(slider, row, 0)

        self.fps_label = QLabel(f'{self.fps} fps')
        self.fps_label.setFixedHeight(20)
        layout.addWidget(self.fps_label, row, 1)
        row += 1

        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setMinimum(1)
        slider.setMaximum(8)
        slider.setValue(self.zoom)
        slider.valueChanged.connect(self.set_zoom_handler)
        layout.addWidget(slider, row, 0)

        self.zoom_label = QLabel(f'{self.zoom} x zoom')
        self.zoom_label.setFixedHeight(20)
        layout.addWidget(self.zoom_label, row, 1)
        row += 1

        layout.setRowMinimumHeight(row, 10)
        row += 1

        self.play_sequence_button = QPushButton('Play sequence')
        self.play_sequence_button.clicked.connect(self.play_sequence_handler)
        layout.addWidget(self.play_sequence_button, row, 0)

        self.play_decoded_sequnce_button = QPushButton('Play decoded sequence')
        self.play_decoded_sequnce_button.clicked.connect(self.play_decoded_sequence_handler)
        layout.addWidget(self.play_decoded_sequnce_button, row, 1)
        row += 1

        self.play_intensity_map_button = QPushButton('Play intensity map')
        self.play_intensity_map_button.clicked.connect(self.play_intensity_map_handler)
        layout.addWidget(self.play_intensity_map_button, row, 0)

        self.play_error_map_button = QPushButton('Play error map')
        self.play_error_map_button.clicked.connect(self.play_error_map_handler)
        layout.addWidget(self.play_error_map_button, row, 1)
        row += 1

        self.play_all_button = QPushButton('Play all')
        self.play_all_button.clicked.connect(self.play_all_handler)
        layout.addWidget(self.play_all_button, row, 0)

        self.show_stats_button = QPushButton('Show stats')
        self.show_stats_button.clicked.connect(self.show_stats_handler)
        layout.addWidget(self.show_stats_button, row, 1)

        return layout

    def prepare_stats_section(self):
        row = 0
        layout = QGridLayout()
        layout.setColumnMinimumWidth(0, 200)
        layout.setColumnMinimumWidth(1, 200)

        tmp = QPushButton('Export stats')
        tmp.clicked.connect(self.export_stats)
        layout.addWidget(tmp, row, 0, 1, 2)

        return layout

    def show_file_dialog_handler(self):
        file = QFileDialog.getOpenFileName(self, 'Choose sequence file', '_sequences', 'YUV (*.yuv)')

        if file and file[0]:
            file_path = file[0]

            name = path.relpath(file_path, start=SEQUENCES_DIR)
            self.sequence_path_label.setText(name)
            self.sequence_path = file_path
            self.update_config_and_buttons_availability()

    def set_fps_handler(self, value):
        self.fps = value
        self.fps_label.setText(f'{value} fps')

    def set_zoom_handler(self, value):
        self.zoom = int(value)
        self.zoom_label.setText(f'{value} x zoom')

    def update_config_and_buttons_availability(self):
        rows = int(self.rows_box.currentText())
        cols = int(self.cols_box.currentText())
        frames = int(self.frames_box.currentText())
        target_bpp = None if self.target_bpp_box.text() in ['', ',', '.'] else float(self.target_bpp_box.text().replace(',', '.'))
        encoding = self.encoding_box.currentData()
        decoding = self.decoding_box.currentData()
        info = self.additional_info_box.text()

        self.config = Config(rows, cols, frames, target_bpp, encoding, decoding, info)

        sequence_exists = path.exists(self.sequence_path)
        decode_sequence_exists = path.exists(decoded_sequence_path(self.sequence_path, self.config.name))
        intensity_map_exists = path.exists(intensity_map_path(self.sequence_path, self.config.name))
        error_map_exists = path.exists(error_map_path(self.sequence_path, self.config.name))
        stats_exist = path.exists(stats_path(self.sequence_path, self.config.name))

        self.play_sequence_button.setDisabled(not sequence_exists)
        self.play_decoded_sequnce_button.setDisabled(not decode_sequence_exists)
        self.play_intensity_map_button.setDisabled(not intensity_map_exists)
        self.play_error_map_button.setDisabled(not error_map_exists)
        self.play_all_button.setDisabled(not (sequence_exists and decode_sequence_exists and intensity_map_exists and error_map_exists))
        self.show_stats_button.setDisabled(not stats_exist)

    def run_codec_handler(self):
        run_codec(self.sequence_path, self.config)

    def play_sequence_handler(self):
        YuvPlayer(self.sequence_path, self.config, True, False, False, False, self.zoom, self.fps).play()

    def play_decoded_sequence_handler(self):
        YuvPlayer(self.sequence_path, self.config, False, True, False, False, self.zoom, self.fps).play()

    def play_intensity_map_handler(self):
        YuvPlayer(self.sequence_path, self.config, False, False, True, False, self.zoom, self.fps).play()

    def play_error_map_handler(self):
        YuvPlayer(self.sequence_path, self.config, False, False, False, True, self.zoom, self.fps).play()

    def play_all_handler(self):
        YuvPlayer(self.sequence_path, self.config, True, True, True, True, self.zoom, self.fps).play()

    def show_stats_handler(self):
        os.startfile(stats_path(self.sequence_path, self.config.name))

    def export_stats(self):
        export_stats_to_excel(RESULTS_DIR)

    def disable_ui(self):
        self.main_widget.setDisabled(True)

    def enable_ui(self):
        self.update_config_and_buttons_availability()
        self.main_widget.setDisabled(False)


class CodecWorker(QRunnable):
    def __init__(self, function, callback, path, data):
        super(CodecWorker, self).__init__()

        self.function = function
        self.callback = callback
        self.path = path
        self.data = data

    @pyqtSlot()
    def run(self):
        self.function(self.path, self.data)
        self.callback()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = App()
    window.show()

    app.exec()
