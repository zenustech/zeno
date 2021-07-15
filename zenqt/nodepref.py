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

        button = QDMGraphicsButton(self)
        M = HORI_MARGIN * 0.2
        H = TEXT_HEIGHT * 0.9
        W = self.width / 2
        rect = QRectF(M, -TEXT_HEIGHT * 2.3, W - M * 2, H)
        button.setGeometry(rect)
        button.setText('new')

        self.height += TEXT_HEIGHT * 1.5

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






