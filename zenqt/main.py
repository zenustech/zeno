from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

import sys

from .window import MainWindow
from .utils import asset_path


def main():
    app = QApplication(sys.argv)
    load_font()
    app.setFont(QFont('Source Sans Pro'))
    win = MainWindow()
    win.show()
    return app.exec_()

def load_font():
    font_file_names = [
        'SourceSansPro-Regular.ttf',
        'SourceSansPro-SemiBold.ttf',
    ]
    for name in font_file_names:
        font_file_path = asset_path(name)
        QFontDatabase().addApplicationFont(font_file_path)
