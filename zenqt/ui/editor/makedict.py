from . import *


class QDMGraphicsSocketEditName(QDMGraphicsSocket):
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

    def setValidator(self, validator):
        self.label.edit.setValidator(validator)

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
        self.socket_keys = ['obj0']
        super().__init__(parent)
        self.dynamic_sockets_is_output = False

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.75

        for key in self.socket_keys:
            socket = QDMGraphicsSocketEditName(self)
            socket.setPos(0, self.height + TEXT_HEIGHT * 0.5)
            socket.setName(key)
            socket.setIsOutput(self.dynamic_sockets_is_output)
            if self.dynamic_sockets_is_output:
                self.outputs[socket.name] = socket
            else:
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


class QDMGraphicsNode_ExtractDict(QDMGraphicsNode_MakeDict):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dynamic_sockets_is_output = True

    def onInputChanged(self):
        pass

    def onOutputChanged(self):
        if len(self.outputs[self.socket_keys[-1]].edges) > 0:
            self.add_new_key()
        else:
            while len(self.socket_keys) > 1 and len(self.outputs[self.socket_keys[-2]].edges) == 0:
                self.del_last_key()
