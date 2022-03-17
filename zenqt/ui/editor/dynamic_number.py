from numpy import vdot
from . import *
from ..visualize import zenvis
from .curve_canvas import ControlPoint
from .frame_curve_editor import CurveWindow

class QDMGraphicsNode_DynamicNumber(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keyframes = {
            'x': [ControlPoint(0, 0)],
            'y': [ControlPoint(0, 0)],
            'z': [ControlPoint(0, 0)],
            'w': [ControlPoint(0, 0)],
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
        button2.callback = self.edit_keyframe
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
                txt += ' {} {} {} {} {} {} {}'.format(
                    p.pos.x,
                    p.pos.y,
                    p.cp_type,
                    p.left_handler.x,
                    p.left_handler.y,
                    p.right_handler.x,
                    p.right_handler.y,
                )
        data['params']['_CONTROL_POINTS'] = txt
        v = self.tmp_value
        data['params']['_TMP'] = '' if self.tmp_value == None else '{} {} {} {}'.format(v[0], v[1], v[2], v[3])
        return ident, data

    def load(self, ident, data):
        if data['params']['_CONTROL_POINTS'] != '':
            txt = data['params']['_CONTROL_POINTS'].split()
            txt = (s for s in txt)
            c = int(next(txt))
            for i in range(c):
                k = next(txt)
                l = int(next(txt))
                ls = []
                for j in range(l):
                    pos_x = int(next(txt))
                    pos_y = float(next(txt))
                    cp = ControlPoint(pos_x, pos_y)
                    cp.cp_type = next(txt)
                    cp.left_handler.x = float(next(txt))
                    cp.left_handler.y = float(next(txt))
                    cp.right_handler.x = float(next(txt))
                    cp.right_handler.y = float(next(txt))
                    ls.append(cp)
                self.keyframes[k] = ls
        return super().load(ident, data)

    def add_keyframe(self):
        f = zenvis.status['target_frame']
        for c in 'xyzw':
            ch = self.keyframes[c]
            ps = list(filter(lambda p: p.pos.x <= f, ch))
            l = len(ps)
            v = self.params[c].getValue()
            if ps[-1].pos.x < f:
                ch.insert(l, ControlPoint(f, v))
            else:
                ch[l-1] = ControlPoint(f, v)
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

    def edit_keyframe(self):
        self.curve_editor = CurveWindow(self.keyframes)
        self.curve_editor.show()
        pass
