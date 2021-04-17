import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .ui import QDMNodeEditorWidget


app = QApplication(sys.argv)
win = QDMNodeEditorWidget()
win.show()
sys.exit(app.exec_())
