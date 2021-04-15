from . import core

import sys

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
 

class OpenGLWidget(QGLWidget):
    def initializeGL(self):
        core.initialize()

    @property
    def curr_frameid(self):
        return core.get_curr_frameid()

    @curr_frameid.setter
    def curr_frameid(self, value):
        return core.set_curr_frameid(value)

    def resizeGL(self, nx, ny):
        print('resize', nx, ny)
        core.set_window_size(nx, ny)

    def paintGL(self):
        core.new_frame()
 
 
if __name__ == "__main__":
	app = QApplication(sys.argv)
	win = MainWindow()
	win.show()
	sys.exit(app.exec_())
