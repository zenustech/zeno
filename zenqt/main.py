from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

import sys

from .window import MainWindow
from .utils import asset_path


def main():
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    font_file_path = asset_path('SourceSansPro-Regular.ttf')
    QFontDatabase().addApplicationFont(font_file_path)
    app.setFont(QFont('Source Sans Pro'))
    win = MainWindow()
    win.show()
    return app.exec_()
