'''
Node Editor UI
'''

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
                | QPainter.SmoothPixmapTransform
                | QPainter.TextAntialiasing)

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
                    if item.isOutput:
                        edge.setSrcPos(item.getCirclePos())
                        edge.setDstPos(pos)
                    else:
                        edge.setSrcPos(pos)
                        edge.setDstPos(item.getCirclePos())
                    item.edges.append(edge)
                    edge.updatePath()
                    self.scene().addItem(edge)
                    self.dragingEdge = edge, item, True

            else:
                item = self.itemAt(event.pos())
                edge, srcItem, preserve = self.dragingEdge
                if isinstance(item, QDMGraphicsSocket):
                    self.addEdge(srcItem, item)
                srcItem.edges.remove(edge)
                self.scene().removeItem(edge)
                self.dragingEdge = None

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragingEdge is not None:
            pos = self.mapToScene(event.pos())
            edge, item, _ = self.dragingEdge
            if item.isOutput:
                edge.setSrcPos(item.getCirclePos())
                edge.setDstPos(pos)
            else:
                edge.setSrcPos(pos)
                edge.setDstPos(item.getCirclePos())
            edge.updatePath()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(0)

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
            return False

        if a.isOutput and not b.isOutput:
            src, dst = a, b
        elif not a.isOutput and b.isOutput:
            src, dst = b, a
        else:
            return False

        edge = QDMGraphicsEdge()
        edge.setSrcSocket(src)
        edge.setDstSocket(dst)
        self.scene().addItem(edge)
        return True


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
            dist = self.dstPos.x() - self.srcPos.x()
            dist = max(100, dist, -dist) * BEZIER_FACTOR
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
        self.label.setDefaultTextColor(QColor('#ffffff'))
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


class QDMGraphicsParam(QGraphicsProxyWidget):
    MARGIN = 10

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initLayout()
        assert hasattr(self, 'layout')

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.widget.setStyleSheet('background-color: #333333; color: #eeeeee')

        self.setWidget(self.widget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setGeometry(QRectF(self.MARGIN, 0, 180 - self.MARGIN * 2, 0))

    def initLayout(self):
        self.edit = QLineEdit()
        self.label = QLabel()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def setLabel(self, label):
        self.label.setText(label)

    def setDefault(self, default):
        raise NotImplementedError


class QDMGraphicsParam_int(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        validator = QIntValidator()
        self.edit.setValidator(validator)

    def setDefault(self, default):
        default = [int(x) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.edit.setText(str(x))
        elif len(default) == 2:
            x, xmin = default
            self.edit.setText(str(x))
            self.validator.setBottom(xmin)
        elif len(default) == 3:
            x, xmin, xmax = default
            self.edit.setText(str(x))
            self.validator.setBottom(xmin)
            self.validator.setTop(xmax)
        else:
            assert False, default


class QDMGraphicsParam_float(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QDoubleValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        default = [float(x) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.edit.setText(str(x))
        elif len(default) == 2:
            x, xmin = default
            self.edit.setText(str(x))
            self.validator.setBottom(xmin)
        elif len(default) == 3:
            x, xmin, xmax = default
            self.edit.setText(str(x))
            self.validator.setBottom(xmin)
            self.validator.setTop(xmax)
        else:
            assert False, default



class QDMGraphicsParam_string(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

    def setDefault(self, default):
        self.edit.setText(default)


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width, self.height = 180, 180
        self.back = TEXT_HEIGHT


        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor('#eeeeee'))
        self.title.setPos(0, -self.back)

        self.params = []
        self.sockets = []

    def initSockets(self, title, inputs=(), outputs=(), params=()):
        self.title.setPlainText(title)

        self.params = []
        for index, (type, label, default) in enumerate(params):
            param = eval('QDMGraphicsParam_' + type)(self)
            param.setLabel(label)
            param.setDefault(default)
            self.params.append(param)

        self.sockets = []
        for index, label in enumerate(inputs):
            socket = QDMGraphicsSocket(self)
            index += len(params)
            y = TEXT_HEIGHT * 0.75 + TEXT_HEIGHT * index
            socket.setPos(0, y)
            socket.setLabel(label)
            socket.setIsOutput(False)
            self.sockets.append(socket)

        for index, label in enumerate(outputs):
            socket = QDMGraphicsSocket(self)
            index += len(params) + len(inputs)
            y = TEXT_HEIGHT * 0.75 + TEXT_HEIGHT * index
            socket.setPos(0, y)
            socket.setLabel(label)
            socket.setIsOutput(True)
            self.sockets.append(socket)

    def boundingRect(self):
        return QRectF(0, -self.back, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        pathContent = QPainterPath()
        pathContent.addRect(0, -self.back, self.width, self.height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#333333'))
        painter.drawPath(pathContent.simplified())

        pathTitle = QPainterPath()
        pathTitle.addRect(0, -self.back, self.width, TEXT_HEIGHT)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#222222'))
        painter.drawPath(pathTitle.simplified())

        pathOutline = QPainterPath()
        pathOutline.addRect(0, -self.back, self.width, self.height)
        pen = QPen(QColor('#cc8844' if self.isSelected() else '#000000'))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(pathOutline.simplified())


class QDMNodeEditorWidget(QWidget):
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
        node1.initSockets('Add',
                ['Input1', 'Input2'],
                ['Output1'],
                [('float', 'VoxelSize', '0.08 0')]
                )
        node1.setPos(-200, -100)
        self.scene.addItem(node1)

        node2 = QDMGraphicsNode()
        node2.initSockets('Sub',
                ['Input1', 'Input2'],
                ['Output1'],
                [('float', 'TimeStep', '0.04 0')]
                )
        node2.setPos(100, 100)
        self.scene.addItem(node2)

        self.layout.addWidget(self.view)
