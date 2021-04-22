'''
Node Editor UI
'''

import json
import random

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .launcher import ZenLauncher


class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        width, height = 6400, 6400
        self.setSceneRect(-width // 2, -height // 2, width, height)
        self.setBackgroundBrush(QColor('#444444'))

        self.descs = {}
        self.cates = {}
        self.nodes = []

    def dumpGraph(self):
        nodes = {}
        for node in self.nodes:
            inputs = {}
            for name, socket in node.inputs.items():
                assert not socket.isOutput
                data = None
                if socket.hasAnyEdge():
                    srcSocket = socket.getTheOnlyEdge().srcSocket
                    data = srcSocket.node.ident, srcSocket.name
                inputs[name] = data

            params = {}
            for name, param in node.params.items():
                value = param.getValue()
                params[name] = value

            uipos = node.pos().x(), node.pos().y()

            data = {
                'name': node.name,
                'inputs': inputs,
                'params': params,
                'uipos': uipos,
            }
            nodes[node.ident] = data

        return nodes

    def newGraph(self):
        for node in list(self.nodes):
            self.removeNode(node)
        self.nodes.clear()

    def loadGraph(self, nodes):
        edges = []
        nodesLut = {}

        for ident, data in nodes.items():
            name = data['name']
            inputs = data['inputs']
            params = data['params']
            posx, posy = data['uipos']

            node = self.makeNode(name)
            node.setIdent(ident)
            node.setName(name)
            node.setPos(posx, posy)

            for name, value in params.items():
                node.params[name].setValue(value)

            for name, input in inputs.items():
                if input is None:
                    continue
                dest = node.inputs[name]
                edges.append((dest, input))

            self.addNode(node)
            nodesLut[ident] = node

        for dest, (ident, name) in edges:
            source = nodesLut[ident].outputs[name]
            self.addEdge(source, dest)

    def addEdge(self, src, dst):
        edge = QDMGraphicsEdge()
        edge.setSrcSocket(src)
        edge.setDstSocket(dst)
        self.addItem(edge)

    def makeNode(self, name):
        desc = self.descs[name]
        node = QDMGraphicsNode()
        node.setName(name)
        node.initSockets(desc.inputs, desc.outputs, desc.params)
        return node

    def addNode(self, node):
        self.nodes.append(node)
        self.addItem(node)

    def removeNode(self, node):
        node.remove()
        self.nodes.remove(node)
        self.removeItem(node)

    def setDescriptors(self, descs):
        self.descs = descs
        for name, desc in descs.items():
            for cate in desc.categories:
                self.cates.setdefault(cate, []).append(name)


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

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.contextMenu)

        self.dragingEdge = None
        self.lastContextMenuPos = None

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
                    if not item.isOutput and len(item.edges):
                        srcItem = item.getTheOnlyEdge().srcSocket
                        item.removeAllEdges()
                        item = srcItem

                    edge = QDMGraphicsPath()
                    pos = self.mapToScene(event.pos())
                    if item.isOutput:
                        edge.setSrcPos(item.getCirclePos())
                        edge.setDstPos(pos)
                    else:
                        edge.setSrcPos(pos)
                        edge.setDstPos(item.getCirclePos())
                    edge.updatePath()
                    self.scene().addItem(edge)
                    self.dragingEdge = edge, item, True

            else:
                item = self.itemAt(event.pos())
                edge, srcItem, preserve = self.dragingEdge
                if isinstance(item, QDMGraphicsSocket):
                    self.addEdge(srcItem, item)
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

    def contextMenu(self, pos):
        menu = QMenu(self)
        acts = []
        for cate_name, type_names in self.scene().cates.items():
            act = QAction()
            act.setText(cate_name)
            childMenu = QMenu()
            childActs = []
            for type_name in type_names:
                childMenu.addAction(type_name)
            act.setMenu(childMenu)
            acts.append(act)
        menu.addActions(acts)
        menu.triggered.connect(self.menuTriggered)
        self.lastContextMenuPos = self.mapToScene(pos)
        menu.exec_(self.mapToGlobal(pos))

    def menuTriggered(self, act):
        name = act.text()
        node = self.scene().makeNode(name)
        node.setPos(self.lastContextMenuPos)
        self.scene().addNode(node)

    def addEdge(self, a, b):
        if a is None or b is None:
            return False

        if a.isOutput and not b.isOutput:
            src, dst = a, b
        elif not a.isOutput and b.isOutput:
            src, dst = b, a
        else:
            return False

        self.scene().addEdge(src, dst)
        return True


TEXT_HEIGHT = 25
HORI_MARGIN = TEXT_HEIGHT * 0.4
SOCKET_RADIUS = TEXT_HEIGHT * 0.3
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
        socket.addEdge(self)
        self.srcSocket = socket

    def setDstSocket(self, socket):
        assert not socket.isOutput
        socket.addEdge(self)
        self.dstSocket = socket

    def updatePosition(self):
        self.srcPos = self.srcSocket.getCirclePos()
        self.dstPos = self.dstSocket.getCirclePos()

    def paint(self, painter, styleOptions, widget=None):
        self.updatePosition()
        self.updatePath()

        super().paint(painter, styleOptions, widget)

    def remove(self):
        if self.srcSocket is not None:
            self.srcSocket.edges.remove(self)
        if self.dstSocket is not None:
            self.dstSocket.edges.remove(self)

        self.scene().removeItem(self)


class QDMGraphicsSocket(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.label = QGraphicsTextItem(self)
        self.label.setDefaultTextColor(QColor('#ffffff'))
        self.label.setPos(SOCKET_RADIUS, -TEXT_HEIGHT / 2)

        self.isOutput = False
        self.edges = set()

        self.node = parent
        self.name = None

    def hasAnyEdge(self):
        return len(self.edges) != 0

    def getTheOnlyEdge(self):
        assert not self.isOutput
        assert len(self.edges) == 1
        return next(iter(self.edges))

    def removeAllEdges(self):
        for edge in list(self.edges):
            edge.remove()
        assert len(self.edges) == 0

    def addEdge(self, edge):
        if not self.isOutput:
            self.removeAllEdges()
        self.edges.add(edge)

    def setIsOutput(self, isOutput):
        self.isOutput = isOutput

    def setName(self, name):
        self.name = name
        self.label.setPlainText(name)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        if self.isOutput:
            return basePos + QPointF(self.node.width, 0)
        else:
            return basePos

    def getCircleBounds(self):
        if self.isOutput:
            return (self.node.width - SOCKET_RADIUS, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)
        else:
            return (-SOCKET_RADIUS, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        painter.setBrush(QColor('#6666cc'))
        pen = QPen(QColor('#111111'))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(*self.getCircleBounds())

    def remove(self):
        for edge in list(self.edges):
            edge.remove()


class QDMGraphicsParam(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initLayout()
        assert hasattr(self, 'layout')

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.widget.setStyleSheet('background-color: #333333; color: #eeeeee')

        self.setWidget(self.widget)
        self.setContentsMargins(0, 0, 0, 0)

        self.name = None

    def initLayout(self):
        self.edit = QLineEdit()
        self.label = QLabel()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def setName(self, name):
        self.name = name
        self.label.setText(name)

    def setDefault(self, default):
        self.setValue(default)

    def getValue(self):
        raise NotImplementedError

    def setValue(self, value):
        self.edit.setText(str(value))


class QDMGraphicsParam_int(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QIntValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        default = [int(x) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.setValue(x)
        elif len(default) == 2:
            x, xmin = default
            self.setValue(x)
            self.validator.setBottom(xmin)
            print(xmin)
        elif len(default) == 3:
            x, xmin, xmax = default
            self.setValue(x)
            self.validator.setBottom(xmin)
            self.validator.setTop(xmax)
        else:
            assert False, default

    def getValue(self):
        return int(self.edit.text())


class QDMGraphicsParam_float(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QDoubleValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        default = [float(x) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.setValue(x)
        elif len(default) == 2:
            x, xmin = default
            self.setValue(x)
            self.validator.setBottom(xmin)
        elif len(default) == 3:
            x, xmin, xmax = default
            self.setValue(x)
            self.validator.setBottom(xmin)
            self.validator.setTop(xmax)
        else:
            assert False, default

    def getValue(self):
        return float(self.edit.text())



class QDMGraphicsParam_string(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

    def getValue(self):
        return str(self.edit.text())


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width = 200
        self.height = 0

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor('#eeeeee'))
        self.title.setPos(HORI_MARGIN * 0.5, -TEXT_HEIGHT)

        self.params = {}
        self.inputs = {}
        self.outputs = {}
        self.name = None
        self.ident = 'No{}'.format(random.randrange(1, 100000))

    def remove(self):
        for socket in list(self.inputs.values()):
            socket.remove()
        for socket in list(self.outputs.values()):
            socket.remove()

    def setIdent(self, ident):
        self.ident = ident

    def setName(self, name):
        self.name = name
        self.title.setPlainText(name)

    def initSockets(self, inputs=(), outputs=(), params=()):
        y = TEXT_HEIGHT * 0.1

        self.params.clear()
        for index, (type, name, defl) in enumerate(params):
            param = eval('QDMGraphicsParam_' + type)(self)
            rect = QRectF(HORI_MARGIN, y, self.width - HORI_MARGIN * 2, 0)
            param.setGeometry(rect)
            param.setName(name)
            param.setDefault(defl)
            self.params[name] = param
            y += param.geometry().height()

        y += TEXT_HEIGHT * 0.5

        self.inputs.clear()
        for index, name in enumerate(inputs):
            socket = QDMGraphicsSocket(self)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(False)
            self.inputs[name] = socket
            y += TEXT_HEIGHT

        self.outputs.clear()
        for index, name in enumerate(outputs):
            socket = QDMGraphicsSocket(self)
            index += len(params) + len(inputs)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(True)
            self.outputs[name] = socket
            y += TEXT_HEIGHT

        y += TEXT_HEIGHT * 0.75
        self.height = y

    def boundingRect(self):
        return QRectF(0, -TEXT_HEIGHT, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        pathContent = QPainterPath()
        pathContent.addRect(0, -TEXT_HEIGHT, self.width, self.height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#333333'))
        painter.drawPath(pathContent.simplified())

        pathTitle = QPainterPath()
        pathTitle.addRect(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#222222'))
        painter.drawPath(pathTitle.simplified())

        pathOutline = QPainterPath()
        pathOutline.addRect(0, -TEXT_HEIGHT, self.width, self.height)
        pen = QPen(QColor('#cc8844' if self.isSelected() else '#000000'))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(pathOutline.simplified())


class QDMFileMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('&File')

        acts = [
                ('&New', QKeySequence.New),
                ('&Open', QKeySequence.Open),
                ('&Save', QKeySequence.Save),
                ('Save &as', QKeySequence.SaveAs),
                (0, 0),
                ('&Execute', QKeySequence('F5')),
                (0, 0),
                ('&Close', QKeySequence.Close),
        ]

        for name, shortcut in acts:
            if not name:
                self.addSeparator()
                continue
            action = QAction(name, self)
            action.setShortcut(shortcut)
            self.addAction(action)


class NodeEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_path = None

        self.setWindowTitle('Node Editor')

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.menubar = QMenuBar()
        self.menu = QDMFileMenu()
        self.menu.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menu)
        self.layout.addWidget(self.menubar)

        self.view = QDMGraphicsView(self)
        self.layout.addWidget(self.view)

        self.launcher = ZenLauncher()
        self.scene = QDMGraphicsScene()
        self.view.setScene(self.scene)

        self.reloadDescriptors()

    def reloadDescriptors(self):
        self.scene.setDescriptors(self.launcher.getDescriptors())

    def menuTriggered(self, act):
        name = act.text()
        if name == '&Close':
            self.close()

        elif name == '&Execute':
            self.do_execute()

        elif name == '&New':
            self.scene.newGraph()

        elif name == '&Open':
            path, kind = QFileDialog.getOpenFileName(self, 'File to Open',
                    '', 'Zensim Graph File(*.zsg);; All Files(*);;')
            if path != '':
                self.do_open(path)
                self.current_path = path

        elif name == 'Save &as' or (name == '&Save' and self.current_path is None):
            path, kind = QFileDialog.getSaveFileName(self, 'Path to Save',
                    '', 'Zensim Graph File(*.zsg);; All Files(*);;')
            if path != '':
                self.do_save(path)
                self.current_path = path

        elif name == '&Save':
            self.do_save(self.current_path)

    def do_execute(self):
        graph = self.scene.dumpGraph()
        self.launcher.launchGraph(graph)

    def do_save(self, path):
        graph = self.scene.dumpGraph()
        with open(path, 'w') as f:
            json.dump(graph, f)

    def do_open(self, path):
        with open(path, 'r') as f:
            graph = json.load(f)
        self.scene.newGraph()
        self.scene.loadGraph(graph)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

        super().keyPressEvent(event)
