from . import *


class QDMGraphicsTextButton(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.node = parent
        self.name = None

        self.initLabel()
        self.setChecked(False)

    def initLabel(self):
        self.label = QGraphicsTextItem(self)
        self.label.setPos(0, - TEXT_HEIGHT * 0.1)
        font = QFont()
        font.setPointSize(style['socket_text_size'])
        self.label.setFont(font)

        document = self.label.document()
        option = document.defaultTextOption()
        option.setAlignment(Qt.AlignCenter)
        document.setDefaultTextOption(option)

    def setText(self, name):
        self.name = name
        self.label.setPlainText(name)

    def getCircleBounds(self):
        return (0, 0, self._width, self._height)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        button_color = 'button_selected_color' if self.checked else 'button_color'
        painter.fillRect(*self.getCircleBounds(), QColor(style[button_color]))

    def setChecked(self, checked):
        self.checked = checked

        text_color = 'button_selected_text_color' if self.checked else 'button_text_color'
        self.label.setDefaultTextColor(QColor(style[text_color]))

    def setWidthHeight(self, width, height):
        self._width = width
        self._height = height
        self.label.setTextWidth(width)

    def on_click(self):
        self.setChecked(not self.checked)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.on_click()

    def setGeometry(self, rect):
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        self.setPos(x, y)
        self.setWidthHeight(w, h)





class QDMGraphicsSocketEdiable(QDMGraphicsSocket):
    def __init__(self, parent=None):
        super().__init__(parent)

    def initLabel(self):
        self.label = QDMGraphicsParam_string(self)
        rect = QRectF(HORI_MARGIN, -TEXT_HEIGHT * 0.5,
            self.node.width - HORI_MARGIN * 3, 0)
        self.label.setGeometry(rect)
        self.label.setPlainText = self.label.setValue
        self.label.getPlainText = self.label.getValue

        self.label.edit.editingFinished.connect(self.name_changed)

    def name_changed(self):
        old_name = self.name
        new_name = self.label.getValue()
        if new_name != old_name:
            self.setName(new_name)
            sockets = self.node.outputs if self.isOutput else self.node.inputs
            if sockets.get(old_name) is self:
                this = sockets.pop(old_name)
                sockets[new_name] = this
            if old_name in self.node.socket_keys:
                idx = self.node.socket_keys.index(old_name)
                self.node.socket_keys[idx] = new_name


class QDMGraphicsNode_MakeDict(QDMGraphicsNode):
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
            self.add_button = QDMGraphicsTextButton(self)
        W = self.width / 4
        rect = QRectF(HORI_MARGIN, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.add_button.setGeometry(rect)
        self.add_button.setText('+')
        self.add_button.on_click = self.add_new_key

        if not hasattr(self, 'del_button'):
            self.del_button = QDMGraphicsTextButton(self)
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


class QDMGraphicsNode_ExtractDict(QDMGraphicsNode):
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
            self.add_button = QDMGraphicsTextButton(self)
        W = self.width / 4
        rect = QRectF(HORI_MARGIN, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.add_button.setGeometry(rect)
        self.add_button.setText('+')
        self.add_button.on_click = self.add_new_key

        if not hasattr(self, 'del_button'):
            self.del_button = QDMGraphicsTextButton(self)
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

