from .editor import *


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

    def showEvent(self, event):
        super().showEvent(event)
        self.scene().trans_x = self.horizontalScrollBar().value()
        self.scene().trans_y = self.verticalScrollBar().value()

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
                    return

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
                    return

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
