import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .ui import ViewportWidget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('zenvis')
        self.resize(1100, 650)
        screen = QDesktopWidget().geometry()
        self_size = self.geometry()
        self.move(
                (screen.width() - self_size.width()) // 2,
                (screen.height() - self_size.height()) // 2)

        #'''
        widget = ViewportWidget()
        self.setCentralWidget(widget)
        '''
        splitter = QSplitter(Qt.Vertical)
        widget = ViewportWidget()
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
        '''

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            app.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
