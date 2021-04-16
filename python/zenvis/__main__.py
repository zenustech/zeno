import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .ui import ViewportWidget
from .ctl import TimelineWidget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('zenvis')
        self.resize(1100, 650)

        scrn_size = QDesktopWidget().geometry()
        self_size = self.geometry()
        self.move(
                (scrn_size.width() - self_size.width()) // 2,
                (scrn_size.height() - self_size.height()) // 2)

        self.viewport = ViewportWidget()
        self.timeline = TimelineWidget()

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.viewport)
        self.splitter.addWidget(self.timeline)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.setCentralWidget(self.splitter)

        self.startTimer(1000 // 60)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            app.exit()

        super().keyPressEvent(event)

    def timerEvent(self, event):
        title = self.viewport.get_status_string()
        self.setWindowTitle(title)

        super().timerEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
