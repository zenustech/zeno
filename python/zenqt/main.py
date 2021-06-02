from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from . import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()
