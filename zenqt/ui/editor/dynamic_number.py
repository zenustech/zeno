from . import *
from ..visualize import zenvis

class QDMGraphicsNode_DynamicNumber(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyframes = {}

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.7

        button = QDMOneClickButton(self)
        button.setPos(50, self.height)
        button.setWidthHeight(100, 20)
        button.setText('Add Keyframe')
        self.height += 40
        button.callback = self.add_keyframe

        button2 = QDMOneClickButton(self)
        button2.setPos(50, self.height)
        button2.setWidthHeight(100, 20)
        button2.setText('Edit Curve')
        self.height += 60

    def dump(self):
        ident, data = super().dump()
        txt = '{}'.format(len(self.keyframes))
        for k, v in self.keyframes.items():
            txt += " {} {} {} {} {}".format(k, v[0], v[1], v[2], v[3])

        data['params']['_POINTS'] = txt
        return ident, data

    def load(self, ident, data):
        return super().load(ident, data)

    def add_keyframe(self):
        v = [
            self.params['x'].getValue(),
            self.params['y'].getValue(),
            self.params['z'].getValue(),
            self.params['w'].getValue(),
        ]
        f = zenvis.status['target_frame']
        self.keyframes[f] = v
