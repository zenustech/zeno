'''
Node Editor UI
'''

import os
import json

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSvg import *

from zenutils import go, gen_unique_ident
import zenapi

from . import asset_path

CURR_VERSION = 'v1'
MAX_STACK_LENGTH = 100

style = {
    'title_color': '#638e77',
    'socket_connect_color': '#638e77',
    'socket_unconnect_color': '#4a4a4a',
    'title_text_color': '#FFFFFF',
    'title_text_size': 10,
    'button_text_size': 10,
    'socket_text_size': 10,
    'param_text_size': 10,
    'socket_text_color': '#FFFFFF',
    'panel_color': '#282828',
    'frame_title_color': '#393939',
    'frame_panel_color': '#1B1B1B',
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
    'text_height': 23,
    'copy_offset_x': 100,
    'copy_offset_y': 100,
    'hori_margin': 9,
    'dummy_socket_offset': 15,
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

        self.nodes = []

        self.history_stack = HistoryStack(self)
        self.moved = False
        self.mmb_press = False
        self.contentChanged = False

        self.scale = 1
        self.trans_x = 0
        self.trans_y = 0

    @property
    def descs(self):
        return self.editor.descs

    @property
    def cates(self):
        return self.editor.cates

    def setContentChanged(self, flag):
        self.contentChanged = flag

    def dumpGraph(self, input_nodes=None):
        nodes = {}
        if input_nodes == None:
            input_nodes = self.nodes
        for node in input_nodes:
            nodes.update(node.dump())
        return nodes

    def newGraph(self):
        for node in list(self.nodes):
            node.remove()
        self.nodes.clear()

    def loadGraphEx(self, graph):
        nodes = graph['nodes']
        self.loadGraph(nodes)
        view = graph['view']
        self.scale = view['scale']
        self.trans_x = view['trans_x']
        self.trans_y = view['trans_y']

    def loadGraph(self, nodes, select_all=False):
        edges = []
        nodesLut = {}

        for ident, data in nodes.items():
            name = data['name']
            if name not in self.descs:
                print('no node class named [{}]'.format(name))
                continue
            node = self.makeNode(name)
            node_edges = node.load(ident, data)
            edges.extend(node_edges)

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

    def makeNodeBase(self, name):
        ctor = globals().get('QDMGraphicsNode_' + name, QDMGraphicsNode)
        node = ctor()
        node.setName(name)
        return node

    def makeNode(self, name):
        node = self.makeNodeBase(name)
        desc = self.descs[name]
        node.desc_inputs = desc['inputs']
        node.desc_outputs = desc['outputs']
        node.desc_params = desc['params']
        return node

    def addNode(self, node):
        self.nodes.append(node)
        self.addItem(node)

    def record(self):
        self.history_stack.record()
        self.setContentChanged(True)

    def undo(self):
        self.history_stack.undo()
        self.setContentChanged(True)

    def redo(self):
        self.history_stack.redo()
        self.setContentChanged(True)

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

        self.node_editor = parent

    def setScene(self, scene):
        super().setScene(scene)
        transform = QTransform()
        transform.scale(scene.scale, scene.scale)
        self.setTransform(transform)
        self.horizontalScrollBar().setValue(scene.trans_x)
        self.verticalScrollBar().setValue(scene.trans_y)

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

    def mouseDoubleClickEvent(self, event):
        itemList = self.scene().selectedItems()
        itemList = [n for n in itemList if isinstance(n, QDMGraphicsNode)]
        if len(itemList) != 1:
            return
        item = itemList[0]
        n = item.name
        if n in self.node_editor.scenes:
            self.node_editor.on_switch_graph(n)

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
                if isinstance(item, QDMGraphicsSocket) and not item.dummy:
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

            self.scene().trans_x = self.horizontalScrollBar().value()
            self.scene().trans_y = self.verticalScrollBar().value()

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
        self.scene().scale = self.transform().m11()

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
        s = self.srcSocket
        if not s.node.collapsed:
            srcPos = s.getCirclePos()
        else:
            srcPos = s.node.dummy_output_socket.getCirclePos()
        s = self.dstSocket
        if not s.node.collapsed:
            dstPos = s.getCirclePos()
        else:
            dstPos = s.node.dummy_input_socket.getCirclePos()
        self.setSrcDstPos(srcPos, dstPos)

        super().updatePath()

    def remove(self):
        if self.srcSocket is not None:
            if self in self.srcSocket.edges:
                self.srcSocket.edges.remove(self)
        if self.dstSocket is not None:
            if self in self.dstSocket.edges:
                self.dstSocket.edges.remove(self)

        self.scene().removeItem(self)


class QDMGraphicsNode_FrameResizeHelper(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.setFlag(QGraphicsItem.ItemIsMovable)

        self.node = parent
        self.name = None

        self.setAcceptHoverEvents(True)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        return basePos

    def getCircleBounds(self):
        return (-SOCKET_RADIUS, -SOCKET_RADIUS,
                2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        painter.setBrush(QColor(style['line_color']))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(*[
            QPointF(0, 0),
            QPointF(10, 0),
            QPointF(0, 10),
        ])

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        p = self.pos()
        self.node.setWidthHeight(p.x(), p.y() + TEXT_HEIGHT)

    def hoverEnterEvent(self, event):
        self.node.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setCursor(Qt.SizeFDiagCursor)

    def hoverLeaveEvent(self, event):
        self.node.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setCursor(Qt.ArrowCursor)


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
        self.dummy = False

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
        if self.hasAnyEdge() or self.dummy:
            socket_color = 'socket_connect_color'
        else:
            socket_color = 'socket_unconnect_color'
        painter.setBrush(QColor(style[socket_color]))
        pen = QPen(QColor(style['line_color']))
        pen.setWidth(style['socket_outline_width'])
        painter.setPen(pen)
        painter.drawEllipse(*self.getCircleBounds())

    def remove(self):
        for edge in list(self.edges):
            edge.remove()


class QDMGraphicsButton(QGraphicsProxyWidget):
    class QDMLabel(QLabel):
        def __init__(self):
            super().__init__()
            font = QFont()
            font.setPointSize(style['button_text_size'])
            self.setFont(font)

        def mousePressEvent(self, event):
            self.on_click()
            super().mousePressEvent(event)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = self.QDMLabel()
        self.widget.setAlignment(Qt.AlignCenter)
        self.widget.on_click = self.on_click
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


class QDMCollapseButton(QSvgWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.render = self.renderer()
        self.load(asset_path('unfold.svg'))
        # PyQt5 >= 5.15
        self.render.setAspectRatioMode(Qt.KeepAspectRatio)

        self.setStyleSheet('background-color: {}'.format(style['title_color']))
        self.node = parent
    
    def isChecked(self):
        return self.collapseds
    
    def mousePressEvent(self, event):
        super().mouseMoveEvent(event)
        self.node.collapsed = not self.node.collapsed
        if self.node.collapsed:
            self.node.collapse()
        else:
            self.node.unfold()

    def update_svg(self):
        if self.node.collapsed:
            self.load(asset_path('collapse.svg'))
        else:
            self.load(asset_path('unfold.svg'))
        self.render.setAspectRatioMode(Qt.KeepAspectRatio)


class QDMGraphicsCollapseButton(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = QDMCollapseButton(parent)
        self.setWidget(self.widget)

    def update_svg(self):
        self.widget.update_svg()

class QDMGraphicsParam(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.initLayout()
        if hasattr(self.edit, 'editingFinished'):
            self.edit.editingFinished.connect(self.edit_finished)
        assert hasattr(self, 'layout')

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.widget.setStyleSheet('background-color: {}; color: #eeeeee'.format(style['panel_color']))

        self.setWidget(self.widget)
        self.setContentsMargins(0, 0, 0, 0)

        self.name = None

    def edit_finished(self):
        self.parent.scene().record()

    def initLayout(self):
        font = QFont()
        font.setPointSize(style['param_text_size'])

        self.edit = QLineEdit()
        self.edit.setFont(font)
        self.label = QLabel()
        self.label.setFont(font)

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
        return str(self.edit.text())

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


class QDMGraphicsParam_multiline_string(QDMGraphicsParam):
    class QDMPlainTextEdit(QPlainTextEdit):
        def focusOutEvent(self, event):
            self.parent.edit_finished()
            super().focusOutEvent(event)

    def initLayout(self):
        font = QFont()
        font.setPointSize(style['param_text_size'])

        self.edit = self.QDMPlainTextEdit()
        self.edit.parent = self
        self.edit.setFont(font)
        self.edit.setStyleSheet('background-color: {}; color: {}'.format(
            style['button_color'], style['button_text_color']))

        self.label = QLabel()
        self.label.setFont(font)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setWidget(self.edit)

    def setGeometry(self, rect):
        rect = QRectF(rect)
        rect.setHeight(6 * TEXT_HEIGHT)
        super().setGeometry(rect)

    def setValue(self, value):
        self.edit.setPlainText(str(value))

    def getValue(self):
        return str(self.edit.toPlainText())

class QDMGraphicsNode_Frame(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setZValue(-2)

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width = style['node_width']
        self.height = 100

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 2, -TEXT_HEIGHT)
        self.title.setTextInteractionFlags(Qt.TextEditorInteraction)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.name = None
        self.ident = None

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().moved = True

    def remove(self):
        self.scene().removeItem(self)

    def setIdent(self, ident):
        self.ident = ident

    def setName(self, name):
        if self.ident is None:
            self.ident = gen_unique_ident(name)
        self.name = name
        self.title.setPlainText(name)

    def initSockets(self):
        self.helper = QDMGraphicsNode_FrameResizeHelper(self)
        h = self.height - TEXT_HEIGHT
        self.helper.setPos(self.width, h)

    def boundingRect(self):
        return QRectF(0, -TEXT_HEIGHT, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        r = style['node_rounded_radius']

        pathContent = QPainterPath()
        rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
        pathContent.addRoundedRect(rect, r, r)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['frame_panel_color']))
        painter.drawPath(pathContent.simplified())

        # title round top
        pathTitle = QPainterPath()
        rect = QRectF(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
        pathTitle.addRoundedRect(rect, r, r)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['frame_title_color']))
        painter.drawPath(pathTitle.simplified())
        
        # title direct bottom
        pathTitle = QPainterPath()
        rect = QRectF(0, -r, self.width, r)
        pathTitle.addRect(rect)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['frame_title_color']))
        painter.drawPath(pathTitle.simplified())

    def setWidthHeight(self, width, height):
        width = max(width, style['node_width'])
        height = max(height, 100)
        self.width = width
        self.height = height
        self.helper.setPos(width, height - TEXT_HEIGHT)

    def dump(self):
        uipos = self.pos().x(), self.pos().y()
        data = {
            'name': self.name,
            'uipos': uipos,
            'width': self.width,
            'height': self.height,
            'title': self.title.toPlainText(),
        }
        return {self.ident: data}
    
    def load(self, ident, data):
        name = data['name']
        posx, posy = data['uipos']

        self.initSockets()
        self.setIdent(ident)
        self.setName(name)
        self.setPos(posx, posy)
        self.setWidthHeight(data['width'], data['height'])

        self.title.setPlainText(data['title'])

        edges = []
        return edges

class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width = style['node_width']
        self.height = 0

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 2, -TEXT_HEIGHT)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.collapse_button = QDMGraphicsCollapseButton(self)
        self.collapse_button.setPos(HORI_MARGIN * 0.5, -TEXT_HEIGHT * 0.84)
        self.collapsed = False

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
        for socket in self.inputs.values():
            for edge in socket.edges:
                edge.updatePath()
        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

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
        collapsed_status = ['collapsed'] if self.collapsed else []
        return [name for name, button in self.options.items() if button.checked] + collapsed_status

    def setOptions(self, options):
        if 'collapsed' in options:
            self.collapse()
        for name, button in self.options.items():
            button.setChecked(name in options)

    def initDummySockets(self):
        h = - TEXT_HEIGHT / 2
        offset = style['dummy_socket_offset']
        s = QDMGraphicsSocket(self)
        s.setPos(-offset, h)
        s.setIsOutput(False)
        s.dummy = True
        self.dummy_input_socket = s
        self.dummy_input_socket.hide()

        w = style['node_width']
        s = QDMGraphicsSocket(self)
        s.setPos(w + offset, h)
        s.setIsOutput(False)
        s.dummy = True
        self.dummy_output_socket = s
        self.dummy_output_socket.hide()

    def initCondButtons(self):
        cond_keys = ['OUT', 'MUTE', 'ONCE', 'VIEW']
        for i, key in enumerate(cond_keys):
            button = QDMGraphicsButton(self)
            M = HORI_MARGIN * 0.2
            H = TEXT_HEIGHT * 0.9
            W = self.width / len(cond_keys)
            rect = QRectF(W * i + M, -TEXT_HEIGHT * 2.3, W - M * 2, H)
            button.setGeometry(rect)
            button.setText(key)
            self.options[key] = button

    def initSockets(self):
        self.initDummySockets()
        self.initCondButtons()

        inputs = self.desc_inputs
        outputs = self.desc_outputs
        params = self.desc_params

        y = self.height + TEXT_HEIGHT * 0.4

        self.params.clear()
        for index, (type, name, defl) in enumerate(params):
            param = globals()['QDMGraphicsParam_' + type](self)
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
            index += len(params) + len(inputs)
            socket.setPos(0, y)
            socket.setName(name)
            socket.setIsOutput(True)
            self.outputs[name] = socket
            y += TEXT_HEIGHT

        y = socket_start + max(len(inputs), len(outputs)) * TEXT_HEIGHT

        y += TEXT_HEIGHT * 0.75
        self.height = y

    def boundingRect(self):
        h = TEXT_HEIGHT if self.collapsed else self.height
        return QRectF(0, -TEXT_HEIGHT, self.width, h).normalized()

    def paint(self, painter, styleOptions, widget=None):
        r = style['node_rounded_radius']

        if not self.collapsed:
            pathContent = QPainterPath()
            rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
            pathContent.addRoundedRect(rect, r, r)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(style['panel_color']))
            painter.drawPath(pathContent.simplified())

            # title round top
            pathTitle = QPainterPath()
            rect = QRectF(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
            pathTitle.addRoundedRect(rect, r, r)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(style['title_color']))
            painter.drawPath(pathTitle.simplified())
            
            # title direct bottom
            pathTitle = QPainterPath()
            rect = QRectF(0, -r, self.width, r)
            pathTitle.addRect(rect)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(style['title_color']))
            painter.drawPath(pathTitle.simplified())

        pathOutline = QPainterPath()
        h = TEXT_HEIGHT if self.collapsed else self.height
        pathOutline.addRoundedRect(0, -TEXT_HEIGHT, self.width, h, r, r)
        pathOutlineColor = 'selected_color' if self.isSelected() else 'line_color'
        pen = QPen(QColor(style[pathOutlineColor]))
        pen.setWidth(style['node_outline_width'])
        painter.setPen(pen)
        if not self.collapsed:
            painter.setBrush(Qt.NoBrush)
        else:
            painter.setBrush(QColor(style['title_color']))
        painter.drawPath(pathOutline.simplified())

    def collapse(self):
        self.dummy_input_socket.show()
        self.dummy_output_socket.show()

        self.collapsed = True
        self.collapse_button.update_svg()
        for v in self.options.values():
            v.hide()
        for v in self.params.values():
            v.hide()
        for v in self.inputs.values():
            v.hide()
        for v in self.outputs.values():
            v.hide()

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

    def unfold(self):
        self.dummy_input_socket.hide()
        self.dummy_output_socket.hide()

        self.collapsed = False
        self.collapse_button.update_svg()
        for v in self.options.values():
            v.show()
        for v in self.params.values():
            v.show()
        for v in self.inputs.values():
            v.show()
        for v in self.outputs.values():
            v.show()

        for socket in self.outputs.values():
            for edge in socket.edges:
                edge.updatePath()

    def dump(self):
        node = self
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
        return {node.ident: data}
    
    def load(self, ident, data):
        node = self
        name = data['name']
        inputs = data['inputs']
        params = data['params']
        posx, posy = data['uipos']
        options = data.get('options', [])

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

        edges = []
        for name, input in inputs.items():
            if input is None:
                continue
            if name not in node.inputs:
                print('no input named [{}] for [{}]'.format(
                    name, nodes[ident]['name']))
                continue
            dest = node.inputs[name]
            edges.append((dest, input))
        return edges


class QDMFileMenu(QMenu):
    def __init__(self):
        super().__init__()

        self.setTitle('&File')

        acts = [
                ('&New', QKeySequence.New),
                ('&Open', QKeySequence.Open),
                ('&Save', QKeySequence.Save),
                ('&Import', 'ctrl+shift+o'),
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
                ('Copy', QKeySequence.Copy),
                ('Paste', QKeySequence.Paste),
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
        self.clipboard = None

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

        self.scenes = {}
        self.descs = {}
        self.cates = {}

        self.initExecute()
        self.initShortcuts()
        self.initDescriptors()

        self.newProgram()
        self.handleEnvironParams()

    def clearScenes(self):
        self.scenes.clear()

    def deleteCurrScene(self):
        for k, v in self.scenes.items():
            if v is self.scene:
                del self.scenes[k]
                break
        self.switchScene('main')

    def switchScene(self, name):
        if name not in self.scenes:
            scene = QDMGraphicsScene()
            scene.editor = self
            scene.record()
            scene.setContentChanged(False)
            self.scenes[name] = scene
        else:
            scene = self.scenes[name]
        self.view.setScene(scene)
        self.edit_graphname.clear()
        self.edit_graphname.addItems(self.scenes.keys())

    @property
    def scene(self):
        return self.view.scene()

    def handleEnvironParams(self):
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

        self.edit_graphname = QComboBox(self)
        self.edit_graphname.setEditable(True)
        self.edit_graphname.move(270, 40)
        self.edit_graphname.resize(130, 30)
        self.edit_graphname.textActivated.connect(self.on_switch_graph)

        self.button_new = QPushButton('New', self)
        self.button_new.move(410, 40)
        self.button_new.resize(80, 30)
        self.button_new.clicked.connect(self.on_new_graph)

        self.button_delete = QPushButton('Delete', self)
        self.button_delete.move(500, 40)
        self.button_delete.resize(80, 30)
        self.button_delete.clicked.connect(self.deleteCurrScene)

    def on_switch_graph(self, name):
        self.switchScene(name)
        self.initDescriptors()
        self.edit_graphname.setCurrentText(name)

    def on_new_graph(self):
        name = self.edit_graphname.currentText()
        self.on_switch_graph(name)
        print('all subgraphs are:', list(self.scenes.keys()))

    def setDescriptors(self, descs):
        self.descs = descs
        self.cates.clear()
        for name, desc in self.descs.items():
            for cate in desc['categories']:
                self.cates.setdefault(cate, []).append(name)

    def initDescriptors(self):
        descs = zenapi.getDescriptors()
        subg_descs = self.getSubgraphDescs()
        descs.update(subg_descs)
        descs.update({
            'Frame': {
                'inputs': [],
                'outputs': [],
                'params': [],
                'categories': ['layout'],
            } 
        })
        self.setDescriptors(descs)

    def on_add(self):
        pos = QPointF(0, 0)
        self.view.contextMenu(pos)

    def on_kill(self):
        zenapi.killProcess()

    def dumpProgram(self):
        graphs = {}
        views = {}
        for name, scene in self.scenes.items():
            nodes = scene.dumpGraph()
            view = {
                'scale': scene.scale,
                'trans_x': scene.trans_x,
                'trans_y': scene.trans_y,
            }
            graphs[name] = {'nodes': nodes, 'view': view}
        prog = {}
        prog['graph'] = graphs
        prog['views'] = views
        prog['descs'] = dict(self.descs)
        prog['version'] = CURR_VERSION
        return prog

    def bkwdCompatProgram(self, prog):
        if 'graph' not in prog:
            if 'main' not in prog:
                prog = {'main': prog}
            prog = {'graph': prog}
        if 'descs' not in prog:
            prog['descs'] = dict(self.descs)
        for name, graph in prog['graph'].items():
            if 'nodes' not in graph:
                prog['graph'][name] = {
                    'nodes': graph,
                    'view': {
                        'scale': 1,
                        'trans_x': 0,
                        'trans_y': 0,
                    },
                }
        if 'version' not in prog:
            prog['version'] = 'v0'
        if prog['version'] != CURR_VERSION:
            print('WARNING: Loading graph of version', prog['version'],
                'with editor version', CURR_VERSION)
        return prog

    def importProgram(self, prog):
        prog = self.bkwdCompatProgram(prog)
        self.setDescriptors(prog['descs'])
        self.scene.newGraph()
        for name, graph in prog['graph'].items():
            if name != 'main':
                print('Importing subgraph', name)
                self.switchScene(name)
                nodes = graph['nodes']
                self.scene.loadGraphEx(graph)
        self.scene.record()
        self.switchScene('main')
        self.initDescriptors()

    def loadProgram(self, prog):
        prog = self.bkwdCompatProgram(prog)
        self.setDescriptors(prog['descs'])
        self.clearScenes()
        for name, graph in prog['graph'].items():
            print('Loading subgraph', name)
            self.switchScene(name)
            self.scene.loadGraphEx(graph)
        self.switchScene('main')
        self.initDescriptors()

    def on_execute(self):
        nframes = int(self.edit_nframes.text())
        prog = self.dumpProgram()
        go(zenapi.launchScene, prog['graph'], nframes)

    def on_delete(self):
        itemList = self.scene.selectedItems()
        if not itemList: return
        for item in itemList:
            if item.scene() is not None:
                item.remove()
        self.scene.record()

    def newProgram(self):
        self.clearScenes()
        self.switchScene('main')

    def getOpenFileName(self):
        path, kind = QFileDialog.getOpenFileName(self, 'File to Open',
                '', 'Zensim Graph File(*.zsg);; All Files(*);;')
        return path

    def getSaveFileName(self):
        path, kind = QFileDialog.getSaveFileName(self, 'Path to Save',
                '', 'Zensim Graph File(*.zsg);; All Files(*);;')
        return path

    def menuTriggered(self, act):
        name = act.text()
        if name == '&New':
            if not self.confirm_discard('New'):
                return
            self.newProgram()
            self.current_path = None

        elif name == '&Open':
            if not self.confirm_discard('Open'):
                return
            path = self.getOpenFileName()
            if path:
                self.do_open(path)
                self.current_path = path

        elif name == 'Save &as' or (name == '&Save' and self.current_path is None):
            path = self.getSaveFileName()
            if path:
                self.do_save(path)
                self.current_path = path

        elif name == '&Save':
            self.do_save(self.current_path)

        elif name == '&Import':
            path = self.getOpenFileName()
            if path != '':
                self.do_import(path)

        elif name == 'Undo':
            self.scene.undo()

        elif name == 'Redo':
            self.scene.redo()

        elif name == 'Copy':
            itemList = self.scene.selectedItems()
            itemList = [n for n in itemList if isinstance(n, QDMGraphicsNode)]
            nodes = self.scene.dumpGraph(itemList)
            self.clipboard = nodes

        elif name == 'Paste':
            if self.clipboard is None:
                return
            itemList = self.scene.selectedItems()
            for i in itemList:
                i.setSelected(False)
            nodes = self.clipboard
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
        prog = self.dumpProgram()
        with open(path, 'w') as f:
            json.dump(prog, f, indent=1)
        for scene in self.scenes.values():
            scene.setContentChanged(False)

    def do_import(self, path):
        with open(path, 'r') as f:
            prog = json.load(f)
        self.importProgram(prog)

    def do_open(self, path):
        with open(path, 'r') as f:
            prog = json.load(f)
        self.loadProgram(prog)

    def confirm_discard(self, title):
        if os.environ.get('ZEN_OPEN'):
            return True
        contentChanged = False
        for scene in self.scenes.values():
            if scene.contentChanged:
                contentChanged = True
        if contentChanged:
            flag = QMessageBox.question(self, title, 'Discard unsaved changes?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return flag == QMessageBox.Yes
        return True

    def getSubgraphDescs(self):
        descs = {}
        for name, scene in self.scenes.items():
            if name == 'main': continue
            graph = scene.dumpGraph()
            subcategory = 'subgraph'
            subinputs = []
            suboutputs = []
            for node in graph.values():
                if node['name'] == 'SubInput':
                    subinputs.append(node['params']['name'])
                elif node['name'] == 'SubOutput':
                    suboutputs.append(node['params']['name'])
                elif node['name'] == 'SubCategory':
                    subcategory = node['params']['name']
            subinputs.extend(self.descs['Subgraph']['inputs'])
            suboutputs.extend(self.descs['Subgraph']['outputs'])
            desc = {}
            desc['inputs'] = subinputs
            desc['outputs'] = suboutputs
            desc['params'] = []
            desc['categories'] = [subcategory]
            descs[name] = desc
        return descs


from .nodepref import *
