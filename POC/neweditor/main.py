from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *



class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        width, height = 64000, 64000
        self.setSceneRect(-width // 2, -height // 2, width, height)
        self.setBackgroundBrush(QColor('#393939'))

        node = QDMGraphicsNode()
        node.addInputSocket()
        node.addInputSocket()
        node.addOutputSocket()
        self.addItem(node)


class QDMGraphicsSocket(QGraphicsItem):
    RADIUS = 14

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self._rect = QRectF(-self.RADIUS, -self.RADIUS,
                2 * self.RADIUS, 2 * self.RADIUS)

    def paint(self, painter, options, widget=None):
        lod = options.levelOfDetailFromTransform(painter.worldTransform())

        pathContent = QPainterPath()
        if lod > 0.25:
            pathContent.addEllipse(self._rect)
        else:
            pathContent.addRect(self._rect)

        painter.setBrush(QColor('#aaaaaa'))
        if self.isSelected():
            pen = QPen()
            pen.setColor(QColor('#66ccff'))
            pen.setWidthF(4 if lod > 0.5 else 8)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.NoPen)

        painter.drawPath(pathContent.simplified())

    def boundingRect(self):
        return self._rect.normalized()


class QDMGraphicsNode(QGraphicsItem):
    WIDTH, HEIGHT = 85, 35
    SOCKET_SPAN_WIDTH = WIDTH * 0.8
    ROUND_RADIUS = 10

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)

        self._rect = QRectF(-self.WIDTH, -self.HEIGHT,
                self.WIDTH * 2, self.HEIGHT * 2)

        self._inputs = []
        self._outputs = []

    def addInputSocket(self):
        socket = QDMGraphicsSocket(self)
        self._inputs.append(socket)
        self._updateInputSocketsPos()
        return socket

    def addOutputSocket(self):
        socket = QDMGraphicsSocket(self)
        self._outputs.append(socket)
        self._updateOutputSocketsPos()
        return socket

    def _updateInputSocketsPos(self):
        for i, socket in enumerate(self._inputs):
            x = (i + 0.5) / len(self._inputs)
            x = 2 * x - 1
            socket.setPos(x * self.WIDTH, -self.HEIGHT)
            print(x)

    def _updateOutputSocketsPos(self):
        for i, socket in enumerate(self._outputs):
            x = (i + 0.5) / len(self._outputs)
            x = 2 * x - 1
            socket.setPos(x * self.WIDTH, self.HEIGHT)
            print(x)

    def paint(self, painter, options, widget=None):
        lod = options.levelOfDetailFromTransform(painter.worldTransform())

        pathContent = QPainterPath()
        if lod > 0.5:
            pathContent.addRoundedRect(self._rect, self.ROUND_RADIUS, self.ROUND_RADIUS)
        else:
            pathContent.addRect(self._rect)

        painter.setBrush(QColor('#884422'))
        if self.isSelected():
            pen = QPen()
            pen.setColor(QColor('#66ccff'))
            pen.setWidthF(4 if lod > 0.5 else 8)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.NoPen)

        painter.drawPath(pathContent.simplified())

    def boundingRect(self):
        return self._rect.normalized()


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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        ZOOM_FACTOR = 1.25
        zoomFactor = 1
        if event.angleDelta().y() > 0:
            zoomFactor = ZOOM_FACTOR
        elif event.angleDelta().y() < 0:
            zoomFactor = 1 / ZOOM_FACTOR

        self.scale(zoomFactor, zoomFactor)


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
    win.setGeometry(0, 0, 950, 1080)
    win.show()
    sys.exit(app.exec_())
