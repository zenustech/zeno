import math
import time
import numpy as np

from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *

import zenvis


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
            self.theta -= dy * math.pi
            self.theta = max(-math.pi / 2, min(self.theta, math.pi / 2))
            self.phi += dx * math.pi
            self.phi %= math.pi * 2

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

    def initializeGL(self):
        zenvis.initializeGL()

    def resizeGL(self, nx, ny):
        self.camera.res = (nx, ny)
        self.camera.update_perspective()

    def paintGL(self):
        zenvis.paintGL()

    def on_update(self):
        self.repaint()


class QDMDisplayMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('Display')

        action = QAction('Show Grid', self)
        action.setCheckable(True)
        action.setChecked(True)
        self.addAction(action)


class DisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.menubar = QMenuBar()
        self.layout.addWidget(self.menubar)

        self.menuDisplay = QDMDisplayMenu()
        self.menuDisplay.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menuDisplay)
        
        self.view = ViewportWidget()
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)
    
    def on_update(self):
        self.view.on_update()
    
    def menuTriggered(self, act):
        if name == 'Show Grid':
            checked = act.isChecked()
            zenvis.pyzenvis.set_show_grid(checked)

for name in ['mousePressEvent', 'mouseMoveEvent', 'wheelEvent']:
    def closure(name):
        oldfunc = getattr(ViewportWidget, name)
        def newfunc(self, event):
            getattr(self.camera, name)(event)
            oldfunc(self, event)
        setattr(ViewportWidget, name, newfunc)
    closure(name)
