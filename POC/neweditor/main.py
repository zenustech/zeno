from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *



class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        palette = QPalette()

        width, height = 64000, 64000
        self.setSceneRect(-width // 2, -height // 2, width, height)
        self.setBackgroundBrush(QColor('#393939'))

        text = self.addText('awesome text')
        text.setDefaultTextColor(QColor('white'))
        text.setFlag(QGraphicsItem.ItemIsSelectable)
        text.setFlag(QGraphicsItem.ItemIsMovable)


class QDMGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(QPainter.Antialiasing
                | QPainter.HighQualityAntialiasing
                | QPainter.SmoothPixmapTransform
                | QPainter.TextAntialiasing)

        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)


class NodeEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        self._view = QDMGraphicsView()
        layout.addWidget(self._view)

        self._scene = QDMGraphicsScene()
        self._view.setScene(self._scene)

        self.setLayout(layout)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor('#393939'))
        palette.setColor(QPalette.WindowText, QColor('white'))
        self.setPalette(palette)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = NodeEditor()
    win.show()
    sys.exit(app.exec_())
