from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

import os

from .visualize.viewport import DisplayWidget
from .visualize.timeline import TimelineWidget
from .editor import NodeEditor

from .utils import asset_path

class EditorTimeline(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.editor = NodeEditor(self, parent)
        self.timeline = TimelineWidget()

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.timeline)
        self.layout.addWidget(self.editor)
        self.setLayout(self.layout)

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitleWithPostfix(None)
        self.setGeometry(0, 0, 1200, 1000)
        #self.setGeometry(0, 0, 800, 1000)

        scrn_size = QDesktopWidget().geometry()
        self_size = self.geometry()
        #self.move(
                #(scrn_size.width() - self_size.width()) // 2,
                #(scrn_size.height() - self_size.height()) // 2)

        self.editorTimeline = EditorTimeline(self)
        if not os.environ.get('ZEN_NOVIEW'):
            self.viewport = DisplayWidget()

        self.timeline = self.editorTimeline.timeline
        self.editor = self.editorTimeline.editor
        self.timeline.setEditor(self.editor)
        self.viewport.timeline = self.timeline

        self.mainsplit = QSplitter(Qt.Vertical)
        if hasattr(self, 'viewport'):
            self.mainsplit.addWidget(self.viewport)
        self.mainsplit.addWidget(self.editorTimeline)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.mainsplit)
        self.setLayout(self.layout)

        self.setWindowIcon(QIcon(asset_path('logo.ico')))

        self.timer = QTimer(self)
        self.timer.start(1000 // 60)
        self.timer.timeout.connect(self.on_update)

    def setWindowTitleWithPostfix(self, postfix):
        #from .. import version
        #title = 'ZENO Qt Editor ({})'.format(version)
        title = 'ZENO Qt Editor'
        if postfix:
            title = '{} - [{}]'.format(title, postfix)
        self.setWindowTitle(title)

    def on_update(self):
        if hasattr(self, 'viewport'):
            self.viewport.on_update()
        self.timeline.on_update()

    def closeEvent(self, event):
        if self.editor.confirm_discard('Exit'):
            event.accept()
        else:
            event.ignore()
