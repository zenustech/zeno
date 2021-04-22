import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .editorui import QDMNodeEditorWidget, QDMGraphicsScene
from .launcher import ZenLauncher


app = QApplication(sys.argv)

lan = ZenLauncher()
win = QDMNodeEditorWidget()
scene = QDMGraphicsScene()
win.setScene(scene)
win.setLauncher(lan)

scene.setDescriptors(lan.getDescriptors())

win.show()
sys.exit(app.exec_())
