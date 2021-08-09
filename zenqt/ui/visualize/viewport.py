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

from . import zenvis


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
        nsamples = os.environ.get('ZEN_MSAA')
        if not nsamples:
            nsamples = 16
        else:
            nsamples = int(nsamples)
        fmt.setSamples(nsamples)
        fmt.setVersion(3, 2)
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

    def paintGL(self):
        if self.record_path:
            old_res = self.camera.res
            self.camera.res = self.record_res
            self.camera.update_perspective()
            zenvis.recordGL(self.record_path)
            self.camera.res = old_res
            self.camera.update_perspective()
        zenvis.paintGL()

    def on_update(self):
        self.updateGL()

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

        action = QAction('Background Color', self)
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
        action.setCheckable(True)
        action.setChecked(False)
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
        self.layout.addWidget(self.view)

    def on_update(self):
        self.view.on_update()

    def menuTriggered(self, act):
        name = act.text()
        if name == 'Show Grid':
            checked = act.isChecked()
            zenvis.status['show_grid'] = checked

        elif name == 'Background Color':
            c = QColor.fromRgbF(*zenvis.core.get_background_color())
            c = QColorDialog.getColor(c)
            if c.isValid():
                zenvis.core.set_background_color(
                    c.redF(),
                    c.greenF(),
                    c.blueF(),
                )


        elif name == 'Record Video':
            checked = act.isChecked()
            self.do_record_video(checked)

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

    def do_record_video(self, checked):
        if checked:
            tmp_path = tempfile.mkdtemp(prefix='recording-')
            assert os.path.isdir(tmp_path)
            self.view.record_path = tmp_path
            self.view.record_res = (1024, 768)
        else:
            tmp_path = self.view.record_path
            assert tmp_path is not None
            self.view.record_path = None
            path = self.get_output_path('.mp4')
            png_paths = os.path.join(tmp_path, '%06d.png')
            cmd = ['ffmpeg', '-r', '60', '-i', png_paths, path]
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
        if os.environ.get('ZEN_NOVIEW'):
            return QSize(1200, 0)
        else:
            return QSize(1200, 400)
