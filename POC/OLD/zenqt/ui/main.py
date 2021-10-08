from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

import sys
import os

from .window import MainWindow
from .utils import asset_path


def main():
    if len(sys.argv) > 1:
        from .system.main import main as _main
        return _main()

    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    if sys.platform == 'win32':
        app = QApplication([sys.argv[0], '-platform', 'windows:dpiawareness=0'])
    elif sys.platform == 'linux':
        app = QApplication([sys.argv[0], '-platform', 'xcb:dpiawareness=0'])
    else:
        app = QApplication([sys.argv[0]])

    font_file_path = asset_path('SourceSansPro-Regular.ttf')
    QFontDatabase().addApplicationFont(font_file_path)
    app.setFont(QFont('Source Sans Pro'))
    win = MainWindow()
    win.show()
    return app.exec_()
