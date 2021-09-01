from . import *

class QDMGraphicsNode_MakeList(QDMGraphicsNode):
    def __init__(self, parent=None):
        self.socket_keys = ['obj0']
        self.socket_key_prefix = 'obj'
        super().__init__(parent)

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.75

        for key in self.socket_keys:
            socket = QDMGraphicsSocket(self)
            socket.setPos(0, self.height + TEXT_HEIGHT * 0.5)
            socket.setName(key)
            socket.setIsOutput(False)
            self.inputs[socket.name] = socket
            self.height += TEXT_HEIGHT
        self.height += TEXT_HEIGHT * 1.5

    def onInputChanged(self):
        if len(self.inputs[self.socket_keys[-1]].edges) > 0:
            self.add_new_key()
        else:
            while len(self.socket_keys) > 1 and len(self.inputs[self.socket_keys[-2]].edges) == 0:
                self.del_last_key()

    def add_new_key(self):
        self.socket_keys.append('{}{}'.format(self.socket_key_prefix, len(self.socket_keys)))
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


class QDMGraphicsNode_ExtractList(QDMGraphicsNode_MakeList):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.socket_keys = ['0']
        self.socket_key_prefix = ''

    def initSockets(self):
        QDMGraphicsNode.initSockets(self)
        self.height -= TEXT_HEIGHT * 0.75

        for key in self.socket_keys:
            socket = QDMGraphicsSocketEditName(self)
            socket.setValidator(QIntValidator())
            socket.setPos(0, self.height + TEXT_HEIGHT * 0.5)
            socket.setName(key)
            socket.setIsOutput(True)
            self.outputs[socket.name] = socket
            self.height += TEXT_HEIGHT
        self.height += TEXT_HEIGHT * 1.5

    def onInputChanged(self):
        pass

    def onOutputChanged(self):
        if len(self.outputs[self.socket_keys[-1]].edges) > 0:
            self.add_new_key()
        else:
            while len(self.socket_keys) > 1 and len(self.outputs[self.socket_keys[-2]].edges) == 0:
                self.del_last_key()
