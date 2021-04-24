import math
import time
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import QGLWidget

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from .coredll import core


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
        core.look_perspective(cx, cy, cz,
                self.theta, self.phi, self.radius,
                self.fov, self.ortho_mode)

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
        super().__init__(parent)

        self.camera = CameraControl()
        self.startTimer(1000 // 60)

    @property
    def res(self):
        return self.camera.res

    @res.setter
    def res(self, value):
        nx, ny = self.camera.res = value
        core.set_window_size(nx, ny)
        self.camera.update_perspective()

    def initializeGL(self):
        core.initialize()

    def resizeGL(self, nx, ny):
        print('resize', nx, ny)
        self.res = nx, ny

    def paintGL(self):
        core.new_frame()

    def timerEvent(self, event):
        self.repaint()
        super().timerEvent(event)


for name in ['mousePressEvent', 'mouseMoveEvent', 'wheelEvent']:
    def closure(name):
        oldfunc = getattr(ViewportWidget, name)
        def newfunc(self, event):
            getattr(self.camera, name)(event)
            oldfunc(self, event)
        setattr(ViewportWidget, name, newfunc)
    closure(name)
