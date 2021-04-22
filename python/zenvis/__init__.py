from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .viewport import ViewportWidget
from .timeline import TimelineWidget

from zenedit import NodeEditor


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('zenvis')
        self.setGeometry(200, 200, 1100, 650)

        scrn_size = QDesktopWidget().geometry()
        self_size = self.geometry()
        self.move(
                (scrn_size.width() - self_size.width()) // 2,
                (scrn_size.height() - self_size.height()) // 2)

        self.viewport = ViewportWidget()
        self.timeline = TimelineWidget()
        self.editor = NodeEditor()

        self.mainsplit = QSplitter(Qt.Horizontal)
        self.mainsplit.setOpaqueResize(True)
        self.mainsplit.addWidget(self.viewport)
        self.mainsplit.addWidget(self.editor)
        self.mainsplit.setStretchFactor(0, 6)
        self.mainsplit.setStretchFactor(1, 3)

        self.central = QWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.mainsplit)
        self.layout.addWidget(self.timeline)
        self.central.setLayout(self.layout)

        self.setCentralWidget(self.central)

        self.startTimer(1000 // 60)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

        super().keyPressEvent(event)

    def timerEvent(self, event):
        title = self.viewport.get_status_string()
        self.setWindowTitle(title)

        super().timerEvent(event)
