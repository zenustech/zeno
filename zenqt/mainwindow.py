from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

from .viewport import ViewportWidget
from .timeline import TimelineWidget
from .editor import NodeEditor

from . import asset_path

class EditorTimeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = NodeEditor()
        self.timeline = TimelineWidget()

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.editor)
        self.layout.addWidget(self.timeline)
        self.setLayout(self.layout)

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('ZENO Qt Editor')
        self.setGeometry(0, 0, 1200, 1000)
        #self.setGeometry(0, 0, 800, 1000)

        scrn_size = QDesktopWidget().geometry()
        self_size = self.geometry()
        #self.move(
                #(scrn_size.width() - self_size.width()) // 2,
                #(scrn_size.height() - self_size.height()) // 2)

        self.editorTimeline = EditorTimeline()
        self.viewport = ViewportWidget()

        self.timeline = self.editorTimeline.timeline
        self.editor = self.editorTimeline.editor
        self.timeline.setEditor(self.editor)

        self.mainsplit = QSplitter(Qt.Vertical)
        self.mainsplit.addWidget(self.viewport)
        self.mainsplit.addWidget(self.editorTimeline)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.mainsplit)
        self.setLayout(self.layout)

        self.setWindowIcon(QIcon(asset_path('logo.ico')))

        self.startTimer(1000 // 60)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

        super().keyPressEvent(event)

    def timerEvent(self, event):
        self.viewport.on_update()
        self.timeline.on_update()
        super().timerEvent(event)

    def closeEvent(self, event):
        if self.editor.confirm_discard('Exit'):
            event.accept()
        else:
            event.ignore()
