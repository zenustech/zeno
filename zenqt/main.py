from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

import sys

from . import MainWindow
from . import asset_path


def main():
    app = QApplication(sys.argv)
    font_file_path = asset_path('SourceSansPro-Regular.ttf')
    QFontDatabase().addApplicationFont(font_file_path)
    font_file_path = asset_path('SourceSansPro-SemiBold.ttf')
    QFontDatabase().addApplicationFont(font_file_path)
    app.setFont(QFont('Source Sans Pro'))
    win = MainWindow()
    win.show()
    return app.exec_()
