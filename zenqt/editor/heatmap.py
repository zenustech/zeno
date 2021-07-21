from . import *


'''
class QDMGraphicsNode_MakeHeatmap(QDMGraphicsNode):
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
'''
