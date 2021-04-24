from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .viewport import ViewportWidget
from .timeline import TimelineWidget

from zenedit import NodeEditor


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('zenvis')
        self.setGeometry(200, 200, 1400, 720)

        scrn_size = QDesktopWidget().geometry()
        self_size = self.geometry()
        self.move(
                (scrn_size.width() - self_size.width()) // 2,
                (scrn_size.height() - self_size.height()) // 2)

        self.editor = NodeEditor()
        self.viewport = ViewportWidget()
        self.timeline = TimelineWidget()

        self.mainsplit = QSplitter(Qt.Horizontal)
        self.mainsplit.setOpaqueResize(True)
        self.mainsplit.addWidget(self.viewport)
        self.mainsplit.addWidget(self.editor)
        self.mainsplit.setStretchFactor(0, 5)
        self.mainsplit.setStretchFactor(1, 2)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.mainsplit)
        self.layout.addWidget(self.timeline)
        self.setLayout(self.layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

        super().keyPressEvent(event)
