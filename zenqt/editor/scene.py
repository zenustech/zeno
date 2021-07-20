from . import *


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


class QDMSearchLineEdit(QLineEdit):
    def __init__(self, menu, view):
        super().__init__(menu)
        self.menu = menu
        self.view = view
        self.wact = QWidgetAction(self.menu)
        self.wact.setDefaultWidget(self)
        self.menu.addAction(self.wact)


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
            ident, data = node.dump()
            nodes[ident] = data
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

        for ident, data in nodes.items():
            name = data['name']
            if name not in self.descs:
                print('no node class named [{}]'.format(name))
                continue
            node = self.makeNode(name)
            node_edges = node.load(ident, data)
            edges.extend(node_edges)
            self.addNode(node)
            if select_all:
                node.setSelected(True)

        """self.loadEdges(edges, select_all)

    def loadEdges(self, edges, select_all=False):"""
        nodesLut = {}

        for node in self.nodes:
            nodesLut[node.ident] = node

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

    def reloadNodes(self):
        print('Reloading all nodes')
        savedNodes = self.dumpGraph()
        self.newGraph()
        self.loadGraph(savedNodes)

    def makeNodeBase(self, name):
        ctor = globals().get('QDMGraphicsNode_' + name, QDMGraphicsNode)
        node = ctor()
        node.setName(name)
        return node

    def makeNode(self, name):
        def myunion(list1, list2):
            import copy
            out = copy.deepcopy(list1)
            if list2 is not None:
                if out is None:
                    out = []
                for e in list2:
                    if e not in out:
                        out.append(e)
            return out

        def myunion2(param1, param2):
            import copy
            out = copy.deepcopy(param1)
            nameList = []
            if out is not None:
                for index, (type, name, defl) in enumerate(out):
                    nameList.append(name)
            if param2 is not None:
                if out is None:
                    out = []
                for index, (type, name, defl) in enumerate(param2):
                    if name not in nameList:
                        out.append((type, name, defl))
            return out

        node = self.makeNodeBase(name)
        node.desc = self.descs[name]
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
        self.needReloadNodes = False

        self.node_editor = parent

    def setScene(self, scene):
        super().setScene(scene)
        transform = QTransform()
        transform.scale(scene.scale, scene.scale)
        self.setTransform(transform)
        self.horizontalScrollBar().setValue(scene.trans_x)
        self.verticalScrollBar().setValue(scene.trans_y)

    def showEvent(self, event):
        super().showEvent(event)
        self.scene().trans_x = self.horizontalScrollBar().value()
        self.scene().trans_y = self.verticalScrollBar().value()

    def updateSearch(self, edit):
        for act in edit.menu.actions():
            if not isinstance(act, QWidgetAction):
                edit.menu.removeAction(act)
        pattern = edit.text()
        keys = self.scene().descs.keys()
        matched = fuzzy_search(pattern, keys)
        for key in matched:
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
        self.autoConnectSpecialNode(node)
        self.scene().record()

    def autoConnectSpecialNode(self, node):
        def connectWith(new_name, sock_name):
            new_node = self.scene().makeNode(new_name)
            new_node.initSockets()
            new_node.setPos(self.lastContextMenuPos + QPointF(300, 0))
            self.scene().addNode(new_node)
            src = node.outputs[sock_name]
            dst = new_node.inputs[sock_name]
            self.addEdge(src, dst)
        if node.name in ['BeginFor', 'BeginForEach']:
            connectWith('EndFor', 'FOR')
        elif node.name == 'FuncBegin':
            connectWith('FuncEnd', 'FUNC')

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
            self.setDragMode(QGraphicsView.NoDrag)

            self.scene().trans_x = self.horizontalScrollBar().value()
            self.scene().trans_y = self.verticalScrollBar().value()

        elif event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)

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


from . import *
