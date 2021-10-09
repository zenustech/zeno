import os
import copy
import time
import numpy as np

from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
# from PySide2.QtOpenGL import *

from . import zenvis
from .dialog import RecordVideoDialog
from .camera_keyframe import CameraKeyframeWidget


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
        self.fov = 45.0
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
        ratio = QApplication.desktop().devicePixelRatio()

        x, y = event.x(), event.y()
        dx, dy = x - self.last_pos[0], y - self.last_pos[1]
        dx *= ratio / self.res[0]
        dy *= ratio / self.res[1]

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
            self.phi += dx * np.pi

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

    def set_keyframe(self, keyframe):
        f = keyframe
        self.center = (f[0], f[1], f[2])
        self.theta = f[3]
        self.phi = f[4]
        self.radius = f[5]
        self.fov = f[6]
        self.ortho_mode = f[7]

class ViewportWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        nsamples = os.environ.get('ZEN_MSAA')
        if not nsamples:
            nsamples = 16
        else:
            nsamples = int(nsamples)
        fmt.setSamples(nsamples)
        fmt.setVersion(3, 0)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        super().__init__(parent)
        self.setFormat(fmt)

        self.camera = CameraControl()
        zenvis.camera_control = self.camera
        self.record_path = None
        self.record_res = None

    def initializeGL(self):
        zenvis.initializeGL()

    def resizeGL(self, nx, ny):
        ratio = QApplication.desktop().devicePixelRatio()
        self.camera.res = (nx * ratio, ny * ratio)
        self.camera.update_perspective()

    def paintGL(self):
        zenvis.paintGL()
        self.check_record()

    def check_record(self):
        f = zenvis.get_curr_frameid()
        if self.record_path and f <= self.frame_end:
            old_res = self.camera.res
            self.camera.res = self.record_res
            self.camera.update_perspective()
            zenvis.recordGL(self.record_path)
            self.camera.res = old_res
            self.camera.update_perspective()
            if f == self.frame_end:
                self.parent_widget.record_video.finish_record()

    def on_update(self):
        self.update()

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

        self.addSeparator()

        action = QAction('Smooth Shading', self)
        action.setCheckable(True)
        action.setChecked(False)
        self.addAction(action)

        action = QAction('Wireframe', self)
        action.setCheckable(True)
        action.setChecked(False)
        self.addAction(action)

        self.addSeparator()

        action = QAction('Camera Keyframe', self)
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

        self.record_video = RecordVideoDialog(self)
        self.camera_keyframe_widget = CameraKeyframeWidget(self)
        zenvis.camera_keyframe = self.camera_keyframe_widget

    def on_update(self):
        self.view.on_update()

    def menuTriggered(self, act):
        name = act.text()
        if name == 'Show Grid':
            checked = act.isChecked()
            zenvis.status['show_grid'] = checked

        elif name == 'Smooth Shading':
            checked = act.isChecked()
            zenvis.core.set_smooth_shading(checked)

        elif name == 'Wireframe':
            checked = act.isChecked()
            zenvis.core.set_render_wireframe(checked)

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
            self.record_video.do_record_video()

        elif name == 'Screenshot':
            self.do_screenshot()

        elif name == 'Camera Keyframe':
            self.camera_keyframe_widget.show()

    def get_output_path(self, extname):
        dir_path = 'outputs'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        file_name += extname
        path = os.path.join(dir_path, file_name)
        return path

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
