'''
Node Editor UI
'''

import os
import json

from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtSvg import *

from zenutils import go, gen_unique_ident
from zeno import launch

from . import asset_path

CURR_VERSION = 'v1'
MAX_STACK_LENGTH = 100

style = {
    'title_color': '#1e1e1e',
    'socket_connect_color': '#638e77',
    'socket_unconnect_color': '#4a4a4a',
    'title_text_color': '#FFFFFF',
    'title_text_size': 10,
    'button_text_size': 10,
    'socket_text_size': 10,
    'param_text_size': 10,
    'socket_text_color': '#787878',
    'panel_color': '#1E1E1E',
    'blackboard_title_color': '#393939',
    'blackboard_panel_color': '#1B1B1B',
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
    'node_width': 240,
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


TEXT_HEIGHT = style['text_height']
HORI_MARGIN = style['hori_margin']
SOCKET_RADIUS = style['socket_radius']
BEZIER_FACTOR = 0.5


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
        # PySide2 >= 5.15
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


def fillRect(painter, rect, color, line_width=None, line_color=None):
    if line_width:
        pen = QPen(QColor(line_color))
        pen.setWidth(line_width)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
    else:
        painter.setPen(Qt.NoPen)

    painter.setBrush(QColor(color))

    pathTitle = QPainterPath()
    pathTitle.addRect(rect)
    painter.drawPath(pathTitle.simplified())


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

    def showEvent(self, event):
        super().showEvent(event)
        self.handleEnvironParams()

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
        descs = launch.getDescriptors()
        subg_descs = self.getSubgraphDescs()
        descs.update(subg_descs)
        descs.update({
            'Blackboard': {
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
        launch.killProcess()

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
        # self.scene.newGraph()
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
        go(launch.launchScene, prog['graph'], nframes)

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


from .editor_edge import *
from .editor_socket import *
from .editor_blackboard import *
from .editor_param import *
from .editor_node import *
from .editor_view import *
