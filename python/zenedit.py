import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        width, height = 6400, 6400
        self.setSceneRect(-width // 2, -height // 2, width, height)
        self.setBackgroundBrush(QColor('#444444'))


class QDMGraphicsView(QGraphicsView):
    ZOOM_FACTOR = 1.25

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(QPainter.Antialiasing
                | QPainter.HighQualityAntialiasing
                | QPainter.TextAntialiasing
                | QPainter.SmoothPixmapTransform)

        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.dragingEdge = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            exit()

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

            releaseEvent = QMouseEvent(QEvent.MouseButtonRelease,
                    event.localPos(), event.screenPos(),
                    Qt.MiddleButton, Qt.NoButton,
                    event.modifiers())
            super().mousePressEvent(releaseEvent)

            fakeEvent = QMouseEvent(event.type(),
                    event.localPos(), event.screenPos(),
                    Qt.LeftButton, event.buttons() | Qt.LeftButton,
                    event.modifiers())
            super().mousePressEvent(fakeEvent)

            return

        elif event.button() == Qt.LeftButton:
            if self.dragingEdge is None:
                item = self.itemAt(event.pos())
                if isinstance(item, QDMGraphicsSocket):
                    edge = QDMGraphicsPath()
                    pos = self.mapToScene(event.pos())
                    edge.setSrcPos(item.getCirclePos())
                    edge.setDstPos(pos)
                    edge.updatePath()
                    self.scene().addItem(edge)
                    self.dragingEdge = edge, item

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragingEdge is not None:
            pos = self.mapToScene(event.pos())
            edge, item = self.dragingEdge
            edge.setDstPos(pos)
            edge.updatePath()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(0)

        elif event.button() == Qt.LeftButton:
            if self.dragingEdge is not None:
                item = self.itemAt(event.pos())
                if isinstance(item, QDMGraphicsSocket):
                    edge, srcItem = self.dragingEdge
                    self.dragingEdge = None
                    self.scene().removeItem(edge)
                    self.addEdge(srcItem, item)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoomFactor = 1
        if event.angleDelta().y() > 0:
            zoomFactor = self.ZOOM_FACTOR
        elif event.angleDelta().y() < 0:
            zoomFactor = 1 / self.ZOOM_FACTOR

        self.scale(zoomFactor, zoomFactor)

    def addEdge(self, a, b):
        if a is None or b is None:
            return

        if a.isOutput and not b.isOutput:
            src, dst = a, b
        elif not a.isOutput and b.isOutput:
            src, dst = b, a
        else:
            return

        edge = QDMGraphicsEdge()
        edge.setSrcSocket(src)
        edge.setDstSocket(dst)
        self.scene().addItem(edge)


TEXT_HEIGHT = 25
BEZIER_FACTOR = 0.5


class QDMGraphicsPath(QGraphicsPathItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(-1)

        self.srcPos = QPointF(0, 0)
        self.dstPos = QPointF(0, 0)

    def setSrcPos(self, pos):
        self.srcPos = pos

    def setDstPos(self, pos):
        self.dstPos = pos

    def paint(self, painter, styleOptions, widget=None):
        pen = QPen(QColor('#cc8844' if self.isSelected() else '#000000'))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def updatePath(self):
        path = QPainterPath(self.srcPos)
        if BEZIER_FACTOR == 0:
            path.lineTo(self.dstPos.x(), self.dstPos.y())
        else:
            dist = max(100, self.dstPos.x() - self.srcPos.x()) * BEZIER_FACTOR
            path.cubicTo(self.srcPos.x() + dist, self.srcPos.y(),
                    self.dstPos.x() - dist, self.dstPos.y(),
                    self.dstPos.x(), self.dstPos.y())
        self.setPath(path)


class QDMGraphicsEdge(QDMGraphicsPath):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.srcSocket = None
        self.dstSocket = None

    def setSrcSocket(self, socket):
        assert socket.isOutput
        socket.edges.append(self)
        self.srcSocket = socket

    def setDstSocket(self, socket):
        assert not socket.isOutput
        self.dstSocket = socket

    def updatePosition(self):
        self.srcPos = self.srcSocket.getCirclePos()
        self.dstPos = self.dstSocket.getCirclePos()

    def paint(self, painter, styleOptions, widget=None):
        self.updatePosition()
        self.updatePath()

        super().paint(painter, styleOptions, widget)


class QDMGraphicsSocket(QGraphicsItem):
    RADIUS = TEXT_HEIGHT // 3

    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QGraphicsTextItem(self)
        self.label.setDefaultTextColor(Qt.white)
        self.label.setPos(self.RADIUS, -TEXT_HEIGHT / 2)

        self.isOutput = False
        self.edges = []

        self.node = parent

    def setIsOutput(self, isOutput):
        self.isOutput = isOutput

    def setLabel(self, label):
        self.label.setPlainText(label)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        if self.isOutput:
            return basePos + QPointF(self.node.width, 0)
        else:
            return basePos

    def getCircleBounds(self):
        if self.isOutput:
            return (self.node.width - self.RADIUS, -self.RADIUS,
                    2 * self.RADIUS, 2 * self.RADIUS)
        else:
            return (-self.RADIUS, -self.RADIUS,
                    2 * self.RADIUS, 2 * self.RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        painter.setBrush(QColor('#6666cc'))
        pen = QPen(QColor('#111111'))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(*self.getCircleBounds())


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width, self.height = 180, 120

        #self.proxyContent = QGraphicsProxyWidget(self)
        #self.proxyContent.setWidget(self.content)

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(Qt.white)

        self.initSockets()

    def setTitle(self, title):
        self.title.setPlainText(title)

    def initSockets(self):
        self.sockets = []
        for index in range(3):
            socket = QDMGraphicsSocket(self)
            y = TEXT_HEIGHT * 1.75 + TEXT_HEIGHT * index
            socket.setPos(0, y)
            socket.setLabel('Input%s' % index)
            self.sockets.append(socket)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        pathContent = QPainterPath()
        pathContent.addRect(0, 0, self.width, self.height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#333333'))
        painter.drawPath(pathContent.simplified())

        pathTitle = QPainterPath()
        pathTitle.addRect(0, 0, self.width, TEXT_HEIGHT)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#222222'))
        painter.drawPath(pathTitle.simplified())

        pathOutline = QPainterPath()
        pathOutline.addRect(0, 0, self.width, self.height)
        pen = QPen(QColor('#cc8844' if self.isSelected() else '#000000'))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(pathOutline.simplified())


## 17.54 HAI
## 20.10 HAO
## 25.17 HAO
## 30.48 HAO
## 32.59 HAO
class NodeEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle('Node Editor')

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.scene = QDMGraphicsScene()
        self.view = QDMGraphicsView(self)
        self.view.setScene(self.scene)

        node1 = QDMGraphicsNode()
        node1.setTitle('P2G_Advector')
        node1.setPos(-200, -100)
        self.scene.addItem(node1)

        node2 = QDMGraphicsNode()
        node2.setTitle('FLIP_Creat')
        node2.setPos(100, 100)
        self.scene.addItem(node2)

        node1.sockets[0].setLabel('Output0')
        node1.sockets[0].setIsOutput(True)

        self.layout.addWidget(self.view)
        self.show()



app = QApplication(sys.argv)
win = NodeEditor()
sys.exit(app.exec_())
