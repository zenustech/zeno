from . import core

import sys
import math

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *


class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)
 
		self.setWindowTitle("zenvis")
		self.resize(1100, 650)
		screen = QDesktopWidget().geometry()
		self_size = self.geometry()
		self.move(
                (screen.width() - self_size.width()) // 2,
                (screen.height() - self_size.height()) // 2)
 
		splitter = QSplitter(Qt.Vertical)
		widget = OpenGLWidget()
		splitter.addWidget(widget)
		testedit = QTextEdit()
		splitter.addWidget(testedit)
		splitter.setStretchFactor(0, 3)
		splitter.setStretchFactor(1, 2)
		splitter_main = QSplitter(Qt.Horizontal)
		textedit_main = QTextEdit()
		splitter_main.addWidget(textedit_main)
		splitter_main.addWidget(splitter)
		splitter_main.setStretchFactor(0, 1)
		splitter_main.setStretchFactor(1, 4)
		self.setCentralWidget(splitter_main)


class CameraControl:
    def __init__(self):
        self.mmb_pressed = False
        self.shift_pressed = False
        self.theta = 0.0
        self.phi = 0.0
        self.last_pos = (0, 0)
        self.ortho_mode = False
        self.fov = 6.0
        self.radius = 6.0
        self.res = (0, 0)

        self.update_perspective()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.ortho_mode = not self.ortho_mode

        self.update_perspective()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.mmb_pressed = True
            self.shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)

        self.last_pos = event.x(), event.y()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.mmb_pressed = False
            self.shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)

    def mouseMoveEvent(self, event):
        x, y = event.x(), event.y()
        dx, dy = x - self.last_pos[0], y - self.last_pos[1]
        dx /= self.res[0]
        dy /= self.res[1]

        if self.mmb_pressed:
            if self.shift_pressed:
                pass
            else:
                self.theta -= dy * math.pi
                self.theta = max(-math.pi / 2, min(self.theta, math.pi / 2))
                self.phi += dx * math.pi
                self.phi %= math.pi * 2

        self.last_pos = x, y

        self.update_perspective()

    def update_perspective(self):
        print(self.theta, self.phi, self.radius, self.fov, self.ortho_mode)
        core.look_perspective(self.theta, self.phi, self.radius,
                self.fov, self.ortho_mode)

    def wheelEvent(self, event):
        dy = event.angleDelta().y()
        if dy > 0:
            self.radius *= 0.89
        elif dy < 0:
            self.radius /= 0.89

        self.update_perspective()



class OpenGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.camera = CameraControl()

    @property
    def res(self):
        return self.camera.res

    @res.setter
    def res(self, value):
        nx, ny = self.camera.res = value
        core.set_window_size(nx, ny)

    def initializeGL(self):
        core.initialize()

        self.startTimer(0)

    def resizeGL(self, nx, ny):
        print('resize', nx, ny)
        self.res = nx, ny

    def paintGL(self):
        import random
        print('new_frame', random.random())
        core.new_frame()

    def timerEvent(self, event):
        self.repaint()

        super().timerEvent(event)
 

for name in ['keyPressEvent', 'mousePressEvent', 'mouseReleaseEvent',
        'mouseMoveEvent', 'wheelEvent']:
    def closure(name):
        oldfunc = getattr(OpenGLWidget, name)
        def newfunc(self, event):
            getattr(self.camera, name)(event)
            oldfunc(self, event)
        setattr(OpenGLWidget, name, newfunc)
    closure(name)
 
 
if __name__ == "__main__":
	app = QApplication(sys.argv)
	win = MainWindow()
	win.show()
	sys.exit(app.exec_())
