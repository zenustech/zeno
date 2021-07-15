from .editor import *


class QDMGraphicsNode_Blackboard(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setZValue(-2)

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width = style['node_width']
        self.height = 150

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 2, -TEXT_HEIGHT)
        self.title.setTextInteractionFlags(Qt.TextEditorInteraction)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.content = QGraphicsTextItem(self)
        self.content.setDefaultTextColor(QColor(style['title_text_color']))
        self.content.setPos(HORI_MARGIN, HORI_MARGIN)
        self.content.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.content.setFont(font)

        self.helper = QDMGraphicsBlackboardResizeHelper(self)
        self.setWidthHeight(self.width, self.height)

        self.name = None
        self.ident = None

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().moved = True

    def remove(self):
        self.scene().nodes.remove(self)
        self.scene().removeItem(self)

    def setIdent(self, ident):
        self.ident = ident

    def setName(self, name):
        if self.ident is None:
            self.ident = gen_unique_ident(name)
        self.name = name
        self.title.setPlainText(name)

    def initSockets(self):
        pass

    def boundingRect(self):
        return QRectF(0, -TEXT_HEIGHT, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        r = style['node_rounded_radius']

        pathContent = QPainterPath()
        rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
        pathContent.addRoundedRect(rect, r, r)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['blackboard_panel_color']))
        painter.drawPath(pathContent.simplified())

        # title round top
        pathTitle = QPainterPath()
        rect = QRectF(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
        pathTitle.addRoundedRect(rect, r, r)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['blackboard_title_color']))
        painter.drawPath(pathTitle.simplified())
        
        # title direct bottom
        pathTitle = QPainterPath()
        rect = QRectF(0, -r, self.width, r)
        pathTitle.addRect(rect)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['blackboard_title_color']))
        painter.drawPath(pathTitle.simplified())

        if self.isSelected():
            pathOutline = QPainterPath()
            rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
            pathOutline.addRoundedRect(rect, r, r)
            pen = QPen(QColor(style['selected_color']))
            pen.setWidth(style['node_outline_width'])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(pathOutline.simplified())

    def setWidthHeight(self, width, height):
        width = max(width, style['node_width'])
        height = max(height, 150)
        self.width = width
        self.height = height
        self.helper.setPos(width, height - TEXT_HEIGHT)

        rect = QRectF(HORI_MARGIN, HORI_MARGIN, self.width - HORI_MARGIN * 2,
            self.height - TEXT_HEIGHT - HORI_MARGIN * 2)
        self.content.setTextWidth(self.width - HORI_MARGIN * 2)

    def dump(self):
        uipos = self.pos().x(), self.pos().y()
        data = {
            'name': self.name,
            'uipos': uipos,
            'special': True,
            'width': self.width,
            'height': self.height,
            'title': self.title.toPlainText(),
            'content': self.content.toPlainText(),
        }
        return self.ident, data
    
    def load(self, ident, data):
        name = data['name']
        posx, posy = data['uipos']

        self.initSockets()
        self.setIdent(ident)
        self.setName(name)
        self.setPos(posx, posy)
        self.setWidthHeight(data['width'], data['height'])

        self.title.setPlainText(data['title'])
        self.content.setPlainText(data['content'])

        edges = []
        return edges



class QDMGraphicsSocketEdiable(QDMGraphicsSocket):
    def __init__(self, parent=None):
        super().__init__(parent)

    def initLabel(self):
        self.label = QDMGraphicsParam_string(self)
        rect = QRectF(HORI_MARGIN, -TEXT_HEIGHT * 0.5,
            self.node.width - HORI_MARGIN * 2, 0)
        self.label.setGeometry(rect)
        self.label.setPlainText = self.label.setValue
        self.label.getPlainText = self.label.getValue

        if hasattr(self.label.edit, 'editingFinished'):
            self.label.edit.textChanged.connect(self.name_changed)

    def name_changed(self):
        old_name = self.name
        new_name = self.label.getValue()
        if new_name != old_name:
            self.setName(new_name)
            sockets = self.node.outputs if self.isOutput else self.node.inputs
            if sockets.get(old_name) is self:
                this = sockets.pop(old_name)
                sockets[new_name] = this


class QDMGraphicsNode_MakeSmallDict(QDMGraphicsNode):
    def __init__(self, parent=None):
        self.socket_keys = []
        super().__init__(parent)

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.75

        for key in self.socket_keys:
            """param = QDMGraphicsParam_string(self)
            rect = QRectF(HORI_MARGIN, self.height,
                self.width - HORI_MARGIN * 2, 0)
            param.setGeometry(rect)
            param.setName('')
            param.setDefault('this is {}'.format(index))
            self.params['name{}'.format(index)] = param"""

            socket = QDMGraphicsSocketEdiable(self)
            socket.setPos(0, self.height + TEXT_HEIGHT * 0.5)
            socket.setName(key)
            socket.setIsOutput(False)
            self.inputs[socket.name] = socket

            self.height += TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 0.4

        if not hasattr(self, 'add_button'):
            self.add_button = QDMGraphicsButton(self)
        W = self.width / 4
        rect = QRectF(HORI_MARGIN, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.add_button.setGeometry(rect)
        self.add_button.setText('+')
        self.add_button.on_click = self.add_new_key

        if not hasattr(self, 'del_button'):
            self.del_button = QDMGraphicsButton(self)
        rect = QRectF(HORI_MARGIN + W, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.del_button.setGeometry(rect)
        self.del_button.setText('-')
        self.del_button.on_click = self.del_last_key
        self.height += TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 1.5

    def add_new_key(self):
        self.socket_keys.append('obj{}'.format(len(self.socket_keys)))
        self.reloadSockets()

    def del_last_key(self):
        if len(self.socket_keys):
            self.socket_keys.pop()
            self.reloadSockets()

    def dump(self):
        ident, data = super().dump()
        data['socket_keys'] = tuple(self.socket_keys)
        data['params']['_KEYS'] = '\n'.join(self.socket_keys)
        return ident, data

    def load(self, ident, data):
        if 'socket_keys' in data:
            self.socket_keys = list(data['socket_keys'])

        return super().load(ident, data)


class QDMGraphicsNode_ExtractSmallDict(QDMGraphicsNode):
    def __init__(self, parent=None):
        self.socket_keys = []
        super().__init__(parent)

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.75

        for key in self.socket_keys:
            """param = QDMGraphicsParam_string(self)
            rect = QRectF(HORI_MARGIN, self.height,
                self.width - HORI_MARGIN * 2, 0)
            param.setGeometry(rect)
            param.setName('')
            param.setDefault('this is {}'.format(index))
            self.params['name{}'.format(index)] = param"""

            socket = QDMGraphicsSocketEdiable(self)
            socket.setPos(0, self.height + TEXT_HEIGHT * 0.5)
            socket.setName(key)
            socket.setIsOutput(True)
            self.outputs[socket.name] = socket

            self.height += TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 0.4

        if not hasattr(self, 'add_button'):
            self.add_button = QDMGraphicsButton(self)
        W = self.width / 4
        rect = QRectF(HORI_MARGIN, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.add_button.setGeometry(rect)
        self.add_button.setText('+')
        self.add_button.on_click = self.add_new_key

        if not hasattr(self, 'del_button'):
            self.del_button = QDMGraphicsButton(self)
        rect = QRectF(HORI_MARGIN + W, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.del_button.setGeometry(rect)
        self.del_button.setText('-')
        self.del_button.on_click = self.del_last_key
        self.height += TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 1.5

    def add_new_key(self):
        self.socket_keys.append('obj{}'.format(len(self.socket_keys)))
        self.reloadSockets()

    def del_last_key(self):
        if len(self.socket_keys):
            self.socket_keys.pop()
            self.reloadSockets()

    def dump(self):
        ident, data = super().dump()
        data['socket_keys'] = tuple(self.socket_keys)
        data['params']['_KEYS'] = '\n'.join(self.socket_keys)
        return ident, data

    def load(self, ident, data):
        if 'socket_keys' in data:
            self.socket_keys = list(data['socket_keys'])

        return super().load(ident, data)










