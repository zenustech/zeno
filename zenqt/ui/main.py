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

    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    app = QApplication(['run.py', '-platform', 'windows:dpiawareness=0'])

    font_file_path = asset_path('SourceSansPro-Regular.ttf')
    QFontDatabase().addApplicationFont(font_file_path)
    app.setFont(QFont('Source Sans Pro'))
    win = MainWindow()
    win.show()
    return app.exec_()
