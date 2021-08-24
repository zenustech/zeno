from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *



class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        width, height = 64000, 64000
        self.setSceneRect(-width // 2, -height // 2, width, height)
        self.setBackgroundBrush(QColor('#393939'))

        node = self.addNode()
        node.addInput()
        node.addInput()
        node.addOutput()
        node.setTitle('vdbsmooth')
        node.setPos(0, -100)

        node = self.addNode()
        node.addInput()
        node.addOutput()
        node.setTitle('convertvdb')
        node.setPos(0, 100)

    def addNode(self):
        node = QDMGraphicsNode()
        self.addItem(node)
        return node

    def addLink(self, from_socket, to_socket):
        print(from_socket, to_socket)


class QDMGraphicsPendingLink(QGraphicsPathItem):
    BEZIER_FACTOR = 0

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setZValue(-1)

        self._srcPos = QPointF(0, 0)
        self._dstPos = QPointF(0, 0)

    def paint(self, painter, options, widget=None):
        lod = options.levelOfDetailFromTransform(painter.worldTransform())

        pen = QPen()
        if self.isSelected():
            pen.setColor(QColor('#ffcc66'))
        else:
            pen.setColor(QColor('#66ccff'))
        pen.setWidthF(4 if lod > 0.5 else 8)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def srcPos(self):
        return self._srcPos

    def setSrcPos(self, pos):
        self._srcPos = pos
        self.onUpdatePath()

    def setDstPos(self, pos):
        self._dstPos = pos
        self.onUpdatePath()

    def dstPos(self):
        return self._dstPos

    def onUpdatePath(self):
        path = QPainterPath(self.srcPos())
        if self.BEZIER_FACTOR == 0:
            path.lineTo(self.dstPos().x(), self.dstPos().y())
        else:
            dist = self.dstPos().y() - self.srcPos().y()
            dist = max(100, dist, -dist) * self.BEZIER_FACTOR
            path.cubicTo(self.srcPos().x(), self.srcPos().y() + dist,
                    self.dstPos().x(), self.dstPos().y() - dist,
                    self.dstPos().x(), self.dstPos().y())
        self.setPath(path)


class QDMGraphicsLink(QDMGraphicsPendingLink):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFlag(QGraphicsItem.ItemIsSelectable)



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

        painter.setBrush(QColor('#ffffff'))
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
    ROUND_RADIUS = 10

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)

        self._rect = QRectF(-self.WIDTH, -self.HEIGHT,
                self.WIDTH * 2, self.HEIGHT * 2)

        self._text = QGraphicsTextItem('', self)
        self._text.setDefaultTextColor(QColor('#ffffff'))
        self._text.setScale(2)
        self._text.setPos(self.WIDTH, -self._text.boundingRect().height())

        self._inputs = []
        self._outputs = []

    def title(self):
        self._text.toPlainText()

    def setTitle(self, title):
        self._text.setPlainText(title)

    def addInput(self):
        socket = QDMGraphicsSocket(self)
        self._inputs.append(socket)
        self.onUpdateInputs()
        return socket

    def addOutput(self):
        socket = QDMGraphicsSocket(self)
        self._outputs.append(socket)
        self.onUpdateOutputs()
        return socket

    def onUpdateInputs(self):
        for i, socket in enumerate(self._inputs):
            x = (i + 0.5) / len(self._inputs)
            x = 2 * x - 1
            socket.setPos(x * self.WIDTH, -self.HEIGHT)

    def onUpdateOutputs(self):
        for i, socket in enumerate(self._outputs):
            x = (i + 0.5) / len(self._outputs)
            x = 2 * x - 1
            socket.setPos(x * self.WIDTH, self.HEIGHT)

    def paint(self, painter, options, widget=None):
        lod = options.levelOfDetailFromTransform(painter.worldTransform())

        pathContent = QPainterPath()
        if lod > 0.25:
            pathContent.addRoundedRect(self._rect, self.ROUND_RADIUS, self.ROUND_RADIUS)
        else:
            pathContent.addRect(self._rect)

        painter.setBrush(QColor('#66aa33'))
        if self.isSelected():
            pen = QPen()
            pen.setColor(QColor('#ffcc66'))
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

        self._pendingLink = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            item = self.itemAt(event.pos())
            if isinstance(item, QDMGraphicsSocket):
                self.onSocketClick(item)

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

    def mouseMoveEvent(self, event):
        if self._pendingLink:
            item = self.itemAt(event.pos())
            if isinstance(item, QDMGraphicsSocket):
                pos = item.scenePos()
                self._pendingLink.setDstPos(pos)
        super().mouseMoveEvent(event)

    def onSocketClick(self, socket):
        if self._pendingLink:
            self.scene().addLink(self._pendingLink._socket, socket)
            self.scene().removeItem(self._pendingLink)
        else:
            link = QDMGraphicsPendingLink()
            pos = socket.scenePos()
            link.setSrcPos(pos)
            link.setDstPos(pos)
            self.scene().addItem(link)
            self._pendingLink = link

    def mouseMoveEvent(self, event):
        if self._pendingLink:
            item = self.itemAt(event.pos())
            if item and isinstance(item, QDMGraphicsSocket):
                pos = item.scenePos()
            else:
                pos = self.mapToScene(event.pos())
            self._pendingLink.setDstPos(pos)
        super().mouseMoveEvent(event)


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
