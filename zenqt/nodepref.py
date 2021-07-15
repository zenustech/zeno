from .editor import *


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
            if self.node.inputs.get(old_name) is self:
                this = self.node.inputs.pop(old_name)
                self.node.inputs[new_name] = this


class QDMGraphicsNode_MakeSmallDict(QDMGraphicsNode):
    def __init__(self, parent=None):
        self.input_keys = []
        super().__init__(parent)

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.75

        for key in self.input_keys:
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

        self.add_button = QDMGraphicsButton(self)
        W = self.width / 4
        rect = QRectF(HORI_MARGIN, self.height,
            W - HORI_MARGIN * 2, TEXT_HEIGHT)
        self.add_button.setGeometry(rect)
        self.add_button.setText('+')
        self.add_button.on_click = self.add_new_key

        self.del_button = QDMGraphicsButton(self)
        rect = QRectF(HORI_MARGIN * 2 + W, self.height,
            W - HORI_MARGIN * 2, TEXT_HEIGHT)
        self.del_button.setGeometry(rect)
        self.del_button.setText('+')
        self.del_button.on_click = self.del_last_key
        self.height += TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 1.5

    def add_new_key(self):
        print('add')

    def del_last_key(self):
        print('del')

    def dump(self):
        idata = super().dump()
        data = idata[self.ident]
        data['input_keys'] = tuple(self.input_keys)
        return idata

    def load(self, ident, data):
        if 'input_keys' in data:
            self.input_keys = list(data['input_keys'])

        """self.input_keys.clear()
        for key in data['inputs'].keys():
            if key not in ['SRC', 'COND']:
                self.input_keys.append(key)"""

        return super().load(ident, data)






