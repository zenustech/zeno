from numpy import vdot
from . import *
from ..visualize import zenvis

class QDMGraphicsNode_DynamicNumber(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyframes = {
            'x': [(0, 0)],
            'y': [(0, 0)],
            'z': [(0, 0)],
            'w': [(0, 0)],
        }
        self.base_value = 1
        self.tmp_value = None

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.7

        button = QDMOneClickButton(self)
        button.setPos(50, self.height)
        button.setWidthHeight(100, 20)
        button.setText('Clear Temp')
        self.height += 40
        button.callback = self.clear_temp

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

        def callback(text):
            self.base_value = float(text)

        self.params['speed'].edit.textActivated.connect(callback)

    def dump(self):
        ident, data = super().dump()
        txt = '{}'.format(len(self.keyframes))
        for k, v in self.keyframes.items():
            txt += ' {}'.format(k)
            txt += ' {}'.format(len(v))
            for p in v:
                txt += ' {} {}'.format(p[0], p[1])
        data['params']['_CONTROL_POINTS'] = txt
        v = self.tmp_value
        data['params']['_TMP'] = '' if self.tmp_value == None else '{} {} {} {}'.format(v[0], v[1], v[2], v[3])
        return ident, data

    def load(self, ident, data):
        return super().load(ident, data)

    def add_keyframe(self):
        f = zenvis.status['target_frame']
        for c in 'xyzw':
            ch = self.keyframes[c]
            ps = list(filter(lambda p: p[0] <= f, ch))
            l = len(ps)
            v = self.params[c].getValue()
            if ps[-1][0] < f:
                ch.insert(l, (f, v))
            else:
                ch[l-1] = (f, v)
        self.tmp_value = None

    def clear_temp(self):
        self.tmp_value = None

    def value_modify(self):
        v = [
            self.params['x'].getValue(),
            self.params['y'].getValue(),
            self.params['z'].getValue(),
            self.params['w'].getValue(),
        ]
        self.tmp_value = v
