import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from . import NodeEditor


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = NodeEditor()
    win.setGeometry(200, 200, 800, 600)
    win.show()
    sys.exit(app.exec_())
