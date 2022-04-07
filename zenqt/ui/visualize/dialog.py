import sys
import os
import shutil
import tempfile
import subprocess

from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *

from ...system import fileio
from . import zenvis


class RecordVideoCancelDialog(QDialog):
    def __init__(self, display):
        super().__init__(display)
        self.display = display
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)

        self.setWindowTitle('Zeno')
        self.initUI()

    def initUI(self):
        msg = QLabel('Recording Screen')
        btn = QPushButton('Cancel')
        btn.clicked.connect(self.btn_callback)

        layout = QVBoxLayout()
        layout.addWidget(msg)
        layout.addWidget(btn)
        self.setLayout(layout)
    
    def btn_callback(self):
        view = self.display.view
        # shutil.rmtree(view.record_path, ignore_errors=True)
        view.record_path = None
        self.close()

class RecordVideoDialog(QDialog):
    def __init__(self, display):
        super().__init__()

        self.setWindowTitle('Record screen')   
        self.display = display 
        self.params = {}
        self.initUI()

    def initUI(self):
        frame_start = QLabel('Frame start:')
        self.frame_start_edit = QSpinBox()
        self.frame_start_edit.setMinimum(0)
        self.frame_start_edit.setValue(0)

        frame_end = QLabel('Frame end:')
        self.frame_end_edit = QSpinBox()
        self.frame_end_edit.setMinimum(0)

        fps = QLabel('FPS:')
        self.fps_edit = QSpinBox()
        self.fps_edit.setMinimum(1)
        self.fps_edit.setValue(30)

        viewport_width = QLabel('Width:')
        self.viewport_width_editor = QLineEdit('1280')

        viewport_height = QLabel('Height:')
        self.viewport_height_editor = QLineEdit('720')

        bit_rate = QLabel('Bit rate:')
        self.bit_rate_editor = QLineEdit('20000')

        presets = QLabel('Presets:')
        res_combo = self.build_res_combobox()

        path_button = QPushButton('Path:')
        path_button.clicked.connect(self.path_button_callback)
        self.path_text = QLineEdit()
        self.path_text.setPlaceholderText('(Option)')

        ok_button = QPushButton('OK')
        cancel_button = QPushButton('Cancel')

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(frame_start, 1, 0)
        grid.addWidget(self.frame_start_edit, 1, 1)

        grid.addWidget(frame_end, 2, 0)
        grid.addWidget(self.frame_end_edit, 2, 1)

        grid.addWidget(fps, 3, 0)
        grid.addWidget(self.fps_edit, 3, 1)

        grid.addWidget(bit_rate, 4, 0)
        grid.addWidget(self.bit_rate_editor, 4, 1)

        grid.addWidget(presets, 5, 0)
        grid.addWidget(res_combo, 5, 1)

        grid.addWidget(viewport_width, 6, 0)
        grid.addWidget(self.viewport_width_editor, 6, 1)

        grid.addWidget(viewport_height, 7, 0)
        grid.addWidget(self.viewport_height_editor, 7, 1)

        grid.addWidget(path_button, 8, 0)
        grid.addWidget(self.path_text, 8, 1)

        grid.addWidget(ok_button, 9, 0)
        grid.addWidget(cancel_button, 9, 1)

        self.setLayout(grid) 

    def setFrameCount(self, frame_count):
        self.frame_end_edit.setMaximum(10000)
        self.frame_start_edit.setMaximum(10000)
        # self.frame_end_edit.setMaximum(frame_count - 1)
        # self.frame_end_edit.setValue(frame_count - 1)

    def accept(self):
        r = self.params
        r['frame_start'] = self.frame_start_edit.value()
        r['frame_end'] = self.frame_end_edit.value()
        r['fps'] = self.fps_edit.value()
        r['bit_rate'] = self.bit_rate_editor.text().strip() + 'k'
        r['width'] = int(self.viewport_width_editor.text())
        r['height'] = int(self.viewport_height_editor.text())
        r['path'] = self.path_text.text()
        super().accept()

    def build_res_combobox(self):
        screen_resolution = {
            '540P': (960, 540),
            '720P': (1280, 720),
            '1080P': (1920, 1080),
            '2K': (2560, 1440),
            '4K': (3840, 2160),
        }
        c = QComboBox()
        c.addItems(screen_resolution.keys())
        def callback(text):
            w, h = screen_resolution[text]
            self.viewport_width_editor.setText(str(w))
            self.viewport_height_editor.setText(str(h))
        c.textActivated.connect(callback)
        c.setCurrentIndex(1)
        return c

    def do_record_video(self):
        display = self.display
        params = self.params

        # count = fileio.getFrameCount()
        # if count == 0:
        #     QMessageBox.information(display, 'Zeno', 'Please do simulation before record video!')
        #     return
        self.setFrameCount(1)
        accept = self.exec()
        # if not accept:
        #     return
        # if params['frame_start'] >= params['frame_end']:
        #     QMessageBox.information(display, 'Zeno', 'Frame strat must be less than frame end!')
        #     return
        # params['frame_end'] = min(count - 1, params['frame_end'])

        

        display.timeline.jump_frame(params['frame_start'])
        display.view.frame_end = params['frame_end']

        if params['path']:
            dir_path = params['path']
        else:
            dir_path = display.get_output_path('.mp4')
        dir_path = dir_path.replace('.', '_') + '_images'
        os.makedirs(dir_path)

        display.view.record_path = dir_path
        display.view.record_res = (params['width'], params['height'])

        display.timeline.stop_play()
        display.view.paintGL()
        display.timeline.start_play()
        display.cancel_dialog = RecordVideoCancelDialog(display)
        display.cancel_dialog.show()

    def finish_record(self):
        display = self.display
        display.cancel_dialog.close()
        params = self.params

        tmp_path = display.view.record_path
        assert tmp_path is not None
        display.view.record_path = None
        l = os.listdir(tmp_path)
        l.sort()
        for i in range(len(l)):
            old_name = l[i]
            new_name = '{:07}.png'.format(i + 1)
            old_path = os.path.join(tmp_path, old_name)
            new_path = os.path.join(tmp_path, new_name)
            os.rename(old_path, new_path)

        if params['path']:
            path = params['path']
        else:
            path = display.get_output_path('.mp4')
        png_paths = os.path.join(tmp_path, '%07d.png')
        cmd = [
            'ffmpeg', '-y',
            '-r', str(params['fps']), 
            '-i', png_paths, 
            '-c:v', 'mpeg4',
            '-b:v', params['bit_rate'],
            path,
        ]
        print('Executing command:', cmd)
        try:
            subprocess.check_call(cmd)
            msg = 'Saved video to {}!'.format(path)
            QMessageBox.information(display, 'Record Video', msg)
        except subprocess.CalledProcessError:
            msg = 'Encoding error!'
            QMessageBox.critical(display, 'Record Video', msg)
        # finally:
        #     shutil.rmtree(tmp_path, ignore_errors=True)

    def path_button_callback(self):
        path, kind = QFileDialog.getSaveFileName(self, 'Path to Save', '', 'MP4(*.mp4);;')
        self.path_text.setText(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RecordVideoDialog(None)
    ex.open()
    print(ex.params)

    sys.exit(app.exec_())
