import os
import copy
import time
import shutil
import tempfile
import subprocess
import numpy as np

from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *

import zenvis
from zeno import fileio
from .dialog import *

class CameraControl:
    '''
    MMB to orbit, Shift+MMB to pan, wheel to zoom
    '''
    def __init__(self):
        self.mmb_pressed = False
        self.theta = 0.0
        self.phi = 0.0
        self.last_pos = (0, 0)
        self.center = (0.0, 0.0, 0.0)
        self.ortho_mode = False
        self.fov = 60.0
        self.radius = 5.0
        self.res = (1, 1)

        self.update_perspective()

    def mousePressEvent(self, event):
        if not (event.buttons() & Qt.MiddleButton):
            return

        self.last_pos = event.x(), event.y()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MiddleButton):
            return

        x, y = event.x(), event.y()
        dx, dy = x - self.last_pos[0], y - self.last_pos[1]
        dx /= self.res[0]
        dy /= self.res[1]

        shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)

        if shift_pressed:
            cos_t = np.cos(self.theta)
            sin_t = np.sin(self.theta)
            cos_p = np.cos(self.phi)
            sin_p = np.sin(self.phi)
            back = np.array([ cos_t * sin_p, sin_t, -cos_t * cos_p])
            up = np.array([-sin_t * sin_p, cos_t, sin_t * cos_p])
            right = np.cross(up, back)
            up = np.cross(back, right)
            right /= np.linalg.norm(right)
            up /= np.linalg.norm(up)
            delta = right * dx + up * dy
            center = np.array(self.center)
            center = center + delta * self.radius
            self.center = tuple(center)
        else:
            self.theta -= dy * np.pi
            self.theta = max(-np.pi / 2, min(self.theta, np.pi / 2))
            self.phi += dx * np.pi
            self.phi %= np.pi * 2

        self.last_pos = x, y

        self.update_perspective()

    def update_perspective(self):
        cx, cy, cz = self.center
        zenvis.status['perspective'] = (cx, cy, cz,
                self.theta, self.phi, self.radius,
                self.fov, self.ortho_mode)
        zenvis.status['resolution'] = self.res

    def wheelEvent(self, event):
        dy = event.angleDelta().y()
        scale = 0.89 if dy >= 0 else 1 / 0.89

        shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)
        if shift_pressed:
            self.fov /= scale
        self.radius *= scale

        self.update_perspective()


class ViewportWidget(QGLWidget):
    def __init__(self, parent=None):
        fmt = QGLFormat()
        fmt.setVersion(3, 0)
        fmt.setProfile(QGLFormat.CoreProfile)
        super().__init__(fmt, parent)

        self.camera = CameraControl()
        self.record_path = None
        self.record_res = None

    def initializeGL(self):
        zenvis.initializeGL()

    def resizeGL(self, nx, ny):
        self.camera.res = (nx, ny)
        self.camera.update_perspective()

    def check_record(self):
        f = zenvis.status['frameid']
        if self.record_path and f <= self.frame_end:
            old_res = self.camera.res
            self.camera.res = self.record_res
            self.camera.update_perspective()
            zenvis.recordGL(self.record_path)
            self.camera.res = old_res
            self.camera.update_perspective()
            if f == self.frame_end:
                self.parent_widget.finish_record()

    def paintGL(self):
        self.check_record()
        zenvis.paintGL()

    def on_update(self):
        self.repaint()

@eval('lambda x: x()')
def _():
    for name in ['mousePressEvent', 'mouseMoveEvent', 'wheelEvent']:
        def closure(name):
            oldfunc = getattr(ViewportWidget, name)
            def newfunc(self, event):
                getattr(self.camera, name)(event)
                oldfunc(self, event)
            setattr(ViewportWidget, name, newfunc)
        closure(name)


class QDMDisplayMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('Display')

        action = QAction('Show Grid', self)
        action.setCheckable(True)
        action.setChecked(True)
        self.addAction(action)

class QDMRecordMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('Record')

        action = QAction('Screenshot', self)
        action.setShortcut(QKeySequence('F12'))
        self.addAction(action)

        action = QAction('Record Video', self)
        action.setShortcut(QKeySequence('Shift+F12'))
        self.addAction(action)


class DisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.menubar = QMenuBar()
        self.menubar.setMaximumHeight(26)
        self.layout.addWidget(self.menubar)

        self.menuDisplay = QDMDisplayMenu()
        self.menuDisplay.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menuDisplay)

        self.recordDisplay = QDMRecordMenu()
        self.recordDisplay.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.recordDisplay)

        self.view = ViewportWidget()
        self.view.parent_widget = self
        self.layout.addWidget(self.view)

    def on_update(self):
        self.view.on_update()

    def menuTriggered(self, act):
        name = act.text()
        if name == 'Show Grid':
            checked = act.isChecked()
            zenvis.status['show_grid'] = checked

        elif name == 'Record Video':
            self.do_record_video()

        elif name == 'Screenshot':
            self.do_screenshot()

    def get_output_path(self, extname):
        dir_path = 'outputs'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        file_name += extname
        path = os.path.join(dir_path, file_name)
        return path

    def do_record_video(self):
        count = fileio.getFrameCount()
        if count == 0:
            QMessageBox.information(self, 'Zeno', 'Please do simulation before record video!')
            return
        self.params = {}
        dialog = RecordVideoDialog(self.params, count)
        accept = dialog.exec()
        if not accept:
            return
        if self.params['frame_start'] >= self.params['frame_end']:
            QMessageBox.information(self, 'Zeno', 'Frame strat must be less than frame end!')
            return
        self.params['frame_end'] = min(count - 1, self.params['frame_end'])

        self.timeline.jump_frame(self.params['frame_start'])
        self.timeline.start_play()
        self.view.frame_end = self.params['frame_end']

        tmp_path = tempfile.mkdtemp(prefix='recording-')
        assert os.path.isdir(tmp_path)
        self.view.record_path = tmp_path
        self.view.record_res = (self.params['width'], self.params['height'])

    def finish_record(self):
        tmp_path = self.view.record_path
        assert tmp_path is not None
        self.view.record_path = None
        l = os.listdir(tmp_path)
        l.sort()
        for i in range(len(l)):
            old_name = l[i]
            new_name = '{:06}.png'.format(i + 1)
            old_path = os.path.join(tmp_path, old_name)
            new_path = os.path.join(tmp_path, new_name)
            os.rename(old_path, new_path)
        path = self.get_output_path('.mp4')
        png_paths = os.path.join(tmp_path, '%06d.png')
        cmd = [
            'ffmpeg', 
            '-r', str(self.params['fps']), 
            '-i', png_paths, 
            '-c:v', 'mpeg4', 
            path
        ]
        print('Executing command:', cmd)
        subprocess.check_call(cmd)
        shutil.rmtree(tmp_path, ignore_errors=True)
        zenvis.status['record_video'] = None
        msg = 'Saved video to {}!'.format(path)
        QMessageBox.information(self, 'Record Video', msg)

    def do_screenshot(self):
        path = self.get_output_path('.png')
        zenvis.core.do_screenshot(path)

        msg = 'Saved screenshot to {}!'.format(path)
        QMessageBox.information(self, 'Screenshot', msg)

    def sizeHint(self):
        return QSize(1200, 400)
