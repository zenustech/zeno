'''
Node Editor UI
'''

import os
import json

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from zenutils import go, gen_unique_ident
import zenapi

MAX_STACK_LENGTH = 100

style = {
    'title_color': '#638e77',
    'socket_connect_color': '#638e77',
    'socket_unconnect_color': '#4a4a4a',
    'title_text_color': '#FFFFFF',
    'title_text_size': 10,
    'socket_text_size': 10,
    'socket_text_color': '#FFFFFF',
    'panel_color': '#282828',
    'line_color': '#B0B0B0',
    'background_color': '#2C2C2C',
    'selected_color': '#EE8844',
    'button_color': '#1e1e1e',
    'button_text_color': '#ffffff',
    'button_selected_color': '#449922',
    'button_selected_text_color': '#333333',
    'output_shift': 1,

    'line_width': 3,
    'node_outline_width': 2,
    'socket_outline_width': 2,
    'node_rounded_radius': 6,
    'socket_radius': 8,
    'node_width': 200,
    'text_height': 25,
    'copy_offset_x': 100,
    'copy_offset_y': 100,
    'hori_margin': 10,
}

class HistoryStack:
    def __init__(self, scene):
        self.scene = scene
        self.init_state()
    
    def init_state(self):
        self.current_pointer = -1
        self.stack = []

    def undo(self):
        # can not undo at stack bottom
        if self.current_pointer == 0:
            return

        self.current_pointer -= 1
        current_scene = self.stack[self.current_pointer]
        self.scene.newGraph()
        self.scene.loadGraph(current_scene)

    def redo(self):
        # can not redo at stack top
        if self.current_pointer == len(self.stack) - 1:
            return

        self.current_pointer += 1
        current_scene = self.stack[self.current_pointer]
        self.scene.newGraph()
        self.scene.loadGraph(current_scene)
    
    def record(self):
        if self.current_pointer != len(self.stack) - 1:
            self.stack = self.stack[:self.current_pointer + 1]

        nodes = self.scene.dumpGraph()
        self.stack.append(nodes)
        self.current_pointer += 1

        # limit the stack length
        if self.current_pointer > MAX_STACK_LENGTH:
            self.stack = self.stack[1:]
            self.current_pointer = MAX_STACK_LENGTH

class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        width, height = 64000, 64000
        self.setSceneRect(-width // 2, -height // 2, width, height)
        self.setBackgroundBrush(QColor(style['background_color']))

        self.descs = {}
        self.cates = {}
        self.nodes = []

        self.history_stack = HistoryStack(self)
        self.moved = False
        self.mmb_press = False

    def dumpGraph(self, input_nodes=None):
        nodes = {}
        if input_nodes == None:
            input_nodes = self.nodes
        for node in input_nodes:
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
            options = node.getOptions()

            data = {
                'name': node.name,
                'inputs': inputs,
                'params': params,
                'uipos': uipos,
                'options': options,
            }
            nodes[node.ident] = data

        return nodes

    def newGraph(self):
        for node in list(self.nodes):
            node.remove()
        self.nodes.clear()

    def loadGraph(self, nodes, select_all=False):
        edges = []
        nodesLut = {}

        for ident, data in nodes.items():
            name = data['name']
            inputs = data['inputs']
            params = data['params']
            posx, posy = data['uipos']
            options = data['options']

            if name not in self.descs:
                print('no node class named [{}]'.format(name))
                continue
            node = self.makeNode(name)
            node.initSockets()
            node.setIdent(ident)
            node.setName(name)
            node.setPos(posx, posy)
            node.setOptions(options)

            for name, value in params.items():
                if name not in node.params:
                    print('no param named [{}] for [{}]'.format(
                        name, nodes[ident]['name']))
                    continue
                param = node.params[name]
                param.setValue(value)

            for name, input in inputs.items():
                if input is None:
                    continue
                if name not in node.inputs:
                    print('no input named [{}] for [{}]'.format(
                        name, nodes[ident]['name']))
                    continue
                dest = node.inputs[name]
                edges.append((dest, input))

            self.addNode(node)
            nodesLut[ident] = node
            if select_all:
                node.setSelected(True)

        for dest, (ident, name) in edges:
            if ident not in nodesLut:
                print('no source node ident [{}] for [{}]'.format(
                    ident, dest.name))
                continue
            srcnode = nodesLut[ident]
            if name not in srcnode.outputs:
                print('no output named [{}] for [{}]'.format(
                    name, nodes[ident]['name']))
                continue
            source = srcnode.outputs[name]
            edge = self.addEdge(source, dest)
            if select_all:
                edge.setSelected(True)

    def addEdge(self, src, dst):
        edge = QDMGraphicsEdge()
        edge.setSrcSocket(src)
        edge.setDstSocket(dst)
        edge.updatePath()
        self.addItem(edge)
        return edge

    def searchNode(self, name):
        for key in self.descs.keys():
            if name.lower() in key.lower():
                yield key

    def makeNode(self, name):
        desc = self.descs[name]
        node = QDMGraphicsNode()
        node.setName(name)
        node.desc_inputs = desc['inputs']
        node.desc_outputs = desc['outputs']
        node.desc_params = desc['params']
        return node

    def addNode(self, node):
        self.nodes.append(node)
        self.addItem(node)

    def setDescriptors(self, descs):
        self.descs = descs
        for name, desc in descs.items():
            for cate in desc['categories']:
                self.cates.setdefault(cate, []).append(name)

    def record(self):
        self.history_stack.record()

    def undo(self):
        self.history_stack.undo()

    def redo(self):
        self.history_stack.redo()

    def mousePressEvent(self, event):
        if self.mmb_press:
            return
        super().mousePressEvent(event)


class QDMSearchLineEdit(QLineEdit):
    def __init__(self, menu, view):
        super().__init__(menu)
        self.menu = menu
        self.view = view
        self.wact = QWidgetAction(self.menu)
        self.wact.setDefaultWidget(self)
        self.menu.addAction(self.wact)


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

    def updateSearch(self, edit):
        for act in edit.menu.actions():
            if not isinstance(act, QWidgetAction):
                edit.menu.removeAction(act)
        pattern = edit.text()
        for key in self.scene().searchNode(pattern):
            edit.menu.addAction(key)

    def getCategoryActions(self):
        cates = self.scene().cates
        if not len(cates):
            act = QAction()
            act.setText('ERROR: no descriptors loaded!')
            act.setEnabled(False)
            return [act]
        acts = []
        for cate_name, type_names in cates.items():
            act = QAction()
            act.setText(cate_name)
            childMenu = QMenu()
            childActs = []
            for type_name in type_names:
                childMenu.addAction(type_name)
            act.setMenu(childMenu)
            acts.append(act)
        return acts

    def contextMenu(self, pos):
        menu = QMenu(self)

        edit = QDMSearchLineEdit(menu, self)
        edit.textChanged.connect(lambda: self.updateSearch(edit))
        edit.setFocus()

        acts = self.getCategoryActions()
        menu.addActions(acts)
        menu.triggered.connect(self.menuTriggered)
        self.lastContextMenuPos = self.mapToScene(pos)
        menu.exec_(self.mapToGlobal(pos))

    def menuTriggered(self, act):
        name = act.text()
        if name == '':
            return
        node = self.scene().makeNode(name)
        node.initSockets()
        node.setPos(self.lastContextMenuPos)
        self.scene().addNode(node)
        self.scene().record()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.scene().mmb_press = True

            releaseEvent = QMouseEvent(QEvent.MouseButtonRelease,
                    event.localPos(), event.screenPos(),
                    Qt.MiddleButton, Qt.NoButton,
                    event.modifiers())
            super().mouseReleaseEvent(releaseEvent)

            fakeEvent = QMouseEvent(event.type(),
                    event.localPos(), event.screenPos(),
                    Qt.LeftButton, event.buttons() | Qt.LeftButton,
                    event.modifiers())
            super().mousePressEvent(fakeEvent)

            return

        elif event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)

            if self.dragingEdge is None:
                item = self.itemAt(event.pos())
                if isinstance(item, QDMGraphicsSocket):
                    if not item.isOutput and len(item.edges):
                        srcItem = item.getTheOnlyEdge().srcSocket
                        item.removeAllEdges()
                        item = srcItem

                    edge = QDMGraphicsTempEdge()
                    pos = self.mapToScene(event.pos())
                    edge.setItem(item)
                    edge.setEndPos(pos)
                    edge.updatePath()
                    self.dragingEdge = edge
                    self.scene().addItem(edge)
                    self.scene().update()

            else:
                item = self.itemAt(event.pos())
                edge = self.dragingEdge
                if isinstance(item, QDMGraphicsSocket):
                    self.addEdge(edge.item, item)
                self.scene().removeItem(edge)
                self.scene().update()
                self.dragingEdge = None
                if isinstance(item, QDMGraphicsSocket):
                    self.scene().record()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragingEdge is not None:
            pos = self.mapToScene(event.pos())
            edge = self.dragingEdge
            edge.setEndPos(pos)
            edge.updatePath()
            self.scene().update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if event.button() == Qt.MiddleButton:
            self.scene().mmb_press = False
            self.setDragMode(0)

        elif event.button() == Qt.LeftButton:
            self.setDragMode(0)

        if self.scene().moved:
            self.scene().record()
            self.scene().moved = False

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

        self.scene().addEdge(src, dst)
        return True

TEXT_HEIGHT = style['text_height']
HORI_MARGIN = style['hori_margin']
SOCKET_RADIUS = style['socket_radius']
BEZIER_FACTOR = 0.5


class QDMGraphicsPath(QGraphicsPathItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(-1)

        self.srcPos = QPointF(0, 0)
        self.dstPos = QPointF(0, 0)

    def setSrcDstPos(self, srcPos, dstPos):
        self.srcPos = srcPos
        self.dstPos = dstPos

    def paint(self, painter, styleOptions, widget=None):
        self.updatePath()

        color = 'selected_color' if self.isSelected() else 'line_color'
        pen = QPen(QColor(style[color]))
        pen.setWidth(style['line_width'])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    '''
    def boundingRect(self):
        x0 = min(self.srcPos.x(), self.dstPos.x())
        y0 = min(self.srcPos.y(), self.dstPos.y())
        x1 = max(self.srcPos.x(), self.dstPos.x())
        y1 = max(self.srcPos.y(), self.dstPos.y())
        return QRectF(x0, y0, x1 - x0, y1 - y0)
    '''

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



class QDMGraphicsTempEdge(QDMGraphicsPath):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.item = None
        self.endPos = None

    def setItem(self, item):
        self.item = item

    def setEndPos(self, pos):
        self.endPos = pos

    def updatePath(self):
        if self.item.isOutput:
            self.setSrcDstPos(self.item.getCirclePos(), self.endPos)
        else:
            self.setSrcDstPos(self.endPos, self.item.getCirclePos())

        super().updatePath()


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

    def updatePath(self):
        srcPos = self.srcSocket.getCirclePos()
        dstPos = self.dstSocket.getCirclePos()
        self.setSrcDstPos(srcPos, dstPos)

        super().updatePath()

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
        self.label.setDefaultTextColor(QColor(style['socket_text_color']))
        self.label.setPos(HORI_MARGIN, -TEXT_HEIGHT * 0.5)
        font = QFont()
        font.setPointSize(style['socket_text_size'])
        self.label.setFont(font)

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

        if isOutput:
            document = self.label.document()
            option = document.defaultTextOption()
            option.setAlignment(Qt.AlignRight)
            document.setDefaultTextOption(option)
            width = self.node.boundingRect().width() - HORI_MARGIN * 2
            self.label.setTextWidth(width)

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
        socket_color = 'socket_connect_color' if self.hasAnyEdge() else 'socket_unconnect_color'
        painter.setBrush(QColor(style[socket_color]))
        pen = QPen(QColor(style['line_color']))
        pen.setWidth(style['socket_outline_width'])
        painter.setPen(pen)
        painter.drawEllipse(*self.getCircleBounds())

    def remove(self):
        for edge in list(self.edges):
            edge.remove()


class QDMGraphicsButton(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = QPushButton()
        self.widget.clicked.connect(self.on_click)
        self.setWidget(self.widget)
        self.setChecked(False)

    def on_click(self):
        self.setChecked(not self.checked)

    def setChecked(self, checked):
        self.checked = checked
        if self.checked:
            self.widget.setStyleSheet('background-color: {}; color: {}'.format(
                style['button_selected_color'], style['button_selected_text_color']))
        else:
            self.widget.setStyleSheet('background-color: {}; color: {}'.format(
                style['button_color'], style['button_text_color']))

    def setText(self, text):
        self.widget.setText(text)


class QDMGraphicsParam(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initLayout()
        self.edit.editingFinished.connect(lambda : parent.scene().record())
        assert hasattr(self, 'layout')

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.widget.setStyleSheet('background-color: {}; color: #eeeeee'.format(style['panel_color']))

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

        self.width = style['node_width']
        self.height = 0

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 0.5, -TEXT_HEIGHT * 0.9)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.params = {}
        self.inputs = {}
        self.outputs = {}
        self.options = {}
        self.name = None
        self.ident = None

        self.desc_inputs = []
        self.desc_outputs = []
        self.desc_params = []

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().moved = True

    def remove(self):
        for socket in list(self.inputs.values()):
            socket.remove()
        for socket in list(self.outputs.values()):
            socket.remove()
        
        self.scene().nodes.remove(self)
        self.scene().removeItem(self)

    def setIdent(self, ident):
        self.ident = ident

    def setName(self, name):
        if self.ident is None:
            self.ident = gen_unique_ident(name)
        self.name = name
        self.title.setPlainText(name)

    def getOptions(self):
        return [name for name, button in self.options.items() if button.checked]

    def setOptions(self, options):
        for name, button in self.options.items():
            button.setChecked(name in options)

    def initSockets(self):
        inputs = self.desc_inputs
        outputs = self.desc_outputs
        params = self.desc_params

        y = TEXT_HEIGHT * 0.4

        self.options['OUT'] = button = QDMGraphicsButton(self)
        rect = QRectF(HORI_MARGIN, y, self.width / 2 - HORI_MARGIN * 1.5, 0)
        button.setGeometry(rect)
        button.setText('OUT')

        self.options['MUTE'] = button = QDMGraphicsButton(self)
        rect = QRectF(HORI_MARGIN * 0.5 + self.width / 2,
            y, self.width / 2 - HORI_MARGIN * 1.5, 0)
        button.setGeometry(rect)
        button.setText('MUTE')

        y += TEXT_HEIGHT * 1.3

        self.params.clear()
        for index, (type, name, defl) in enumerate(params):
            param = eval('QDMGraphicsParam_' + type)(self)
            rect = QRectF(HORI_MARGIN, y, self.width - HORI_MARGIN * 2, 0)
            param.setGeometry(rect)
            param.setName(name)
            param.setDefault(defl)
            self.params[name] = param
            y += param.geometry().height()

        if len(params):
            y += TEXT_HEIGHT * 0.7
        else:
            y += TEXT_HEIGHT * 0.4

        socket_start = y + TEXT_HEIGHT * style['output_shift']

        self.inputs.clear()
        for index, name in enumerate(inputs):
            socket = QDMGraphicsSocket(self)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(False)
            self.inputs[name] = socket
            y += TEXT_HEIGHT

        y = socket_start
        if len(inputs) > len(outputs):
            y += (len(inputs) - len(outputs)) * TEXT_HEIGHT

        self.outputs.clear()
        for index, name in enumerate(outputs):
            socket = QDMGraphicsSocket(self)
            index += len(self.desc_params) + len(self.desc_inputs)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(True)
            self.outputs[name] = socket
            y += TEXT_HEIGHT

        y = socket_start + max(len(inputs), len(outputs)) * TEXT_HEIGHT

        y += TEXT_HEIGHT * 0.75
        self.height = y

    def boundingRect(self):
        return QRectF(0, -TEXT_HEIGHT, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        pathContent = QPainterPath()
        pathContent.addRect(0, -TEXT_HEIGHT, self.width, self.height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['panel_color']))
        painter.drawPath(pathContent.simplified())

        pathTitle = QPainterPath()
        pathTitle.addRect(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['title_color']))
        painter.drawPath(pathTitle.simplified())

        pathOutline = QPainterPath()
        r = style['node_rounded_radius']
        pathOutline.addRoundedRect(0, -TEXT_HEIGHT, self.width, self.height, r, r)
        pathOutlineColor = 'selected_color' if self.isSelected() else 'line_color'
        pen = QPen(QColor(style[pathOutlineColor]))
        pen.setWidth(style['node_outline_width'])
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
        ]

        for name, shortcut in acts:
            if not name:
                self.addSeparator()
                continue
            action = QAction(name, self)
            action.setShortcut(shortcut)
            self.addAction(action)

class QDMEditMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('&Edit')

        acts = [
                ('Undo', QKeySequence.Undo),
                ('Redo', QKeySequence.Redo),
                (None, None),
                ('Duplicate', QKeySequence.Copy),
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

        self.menuEdit = QDMEditMenu()
        self.menuEdit.triggered.connect(self.menuTriggered)
        self.menubar.addMenu(self.menuEdit)
        self.layout.addWidget(self.menubar)

        self.view = QDMGraphicsView(self)
        self.layout.addWidget(self.view)

        self.scene = QDMGraphicsScene()
        self.scene.record()
        self.view.setScene(self.scene)

        self.initExecute()
        self.initShortcuts()
        self.refreshDescriptors()

        if os.environ.get('ZEN_OPEN'):
            path = os.environ['ZEN_OPEN']
            self.do_open(path)
            self.current_path = path

        if os.environ.get('ZEN_DOEXEC'):
            print('ZEN_DOEXEC found, direct execute')
            self.on_execute()

    def initShortcuts(self):
        self.msgF5 = QShortcut(QKeySequence('F5'), self)
        self.msgF5.activated.connect(self.on_execute)

        self.msgDel = QShortcut(QKeySequence('Del'), self)
        self.msgDel.activated.connect(self.on_delete)

    def initExecute(self):
        validator = QIntValidator()
        validator.setBottom(0)
        self.edit_nframes = QLineEdit(self)
        self.edit_nframes.setValidator(validator)
        self.edit_nframes.move(20, 40)
        self.edit_nframes.resize(30, 30)
        self.edit_nframes.setText('1')

        self.button_execute = QPushButton('Execute', self)
        self.button_execute.move(60, 40)
        self.button_execute.resize(90, 30)
        self.button_execute.clicked.connect(self.on_execute) 

        self.button_kill = QPushButton('Kill', self)
        self.button_kill.move(160, 40)
        self.button_kill.resize(80, 30)
        self.button_kill.clicked.connect(self.on_kill) 

    def refreshDescriptors(self):
        self.scene.setDescriptors(zenapi.getDescriptors())

    def on_add(self):
        pos = QPointF(0, 0)
        self.view.contextMenu(pos)

    def on_connect(self):
        baseurl = self.edit_baseurl.text()
        import zenwebcfg
        zenwebcfg.connectServer(baseurl)
        self.refreshDescriptors()

    def on_kill(self):
        zenapi.killProcess()

    def on_execute(self):
        nframes = int(self.edit_nframes.text())
        graph = self.scene.dumpGraph()
        go(zenapi.launchGraph, graph, nframes)

    def on_delete(self):
        itemList = self.scene.selectedItems()
        if not itemList: return
        for item in itemList:
            item.remove()
        self.scene.record()

    def menuTriggered(self, act):
        name = act.text()
        if name == '&New':
            self.scene.newGraph()
            self.scene.history_stack.init_state()
            self.scene.record()
            self.current_path = None

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

        elif name == 'Undo':
            self.scene.undo()

        elif name == 'Redo':
            self.scene.redo()

        elif name == 'Duplicate':
            itemList = self.scene.selectedItems()
            for i in itemList:
                i.setSelected(False)
            itemList = [n for n in itemList if isinstance(n, QDMGraphicsNode)]
            nodes = self.scene.dumpGraph(itemList)
            nid_map = {}
            for nid in nodes:
                nid_map[nid] = gen_unique_ident()
            new_nodes = {}
            for nid, n in nodes.items():
                x, y = n['uipos']
                n['uipos'] = (x + style['copy_offset_x'], y + style['copy_offset_y'])
                inputs = n['inputs']
                for name, info in inputs.items():
                    if info == None:
                        continue
                    nid_, name_ = info
                    if nid_ in nid_map:
                        info = (nid_map[nid_], name_)
                    else:
                        info = None
                    inputs[name] = info
                new_nodes[nid_map[nid]] = n
            self.scene.loadGraph(new_nodes, select_all=True)
            self.scene.record()

    def do_save(self, path):
        graph = self.scene.dumpGraph()
        with open(path, 'w') as f:
            json.dump(graph, f)

    def do_open(self, path):
        with open(path, 'r') as f:
            graph = json.load(f)
        self.scene.newGraph()
        self.scene.history_stack.init_state()
        self.scene.loadGraph(graph)
        self.scene.record()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

        super().keyPressEvent(event)
