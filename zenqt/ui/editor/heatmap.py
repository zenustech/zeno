from . import *


class QDMGraphicsNode_MakeHeatmap(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_ramps = [
                [0.0, (0, 0, 0)],
                [0.5, (1, 0, 0)],
                [0.9, (1, 1, 0)],
                [1.0, (1, 1, 1)],
        ]

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.7

        button = QDMOneClickButton(self)
        button.callback = self.open_window
        button.setPos(50, self.height)
        button.setWidthHeight(100, 20)
        button.setText('Edit')
        self.height += 60

    def dump(self):
        ident, data = super().dump()
        ramps = tuple(sorted(self.color_ramps, key=lambda x: x[0]))
        data['color_ramps'] = tuple(ramps)
        data['params']['_RAMPS'] = f'{len(ramps)}' + ''.join(
                f'\n{f} {r} {g} {b}' for f, (r, g, b) in ramps)
        return ident, data

    def load(self, ident, data):
        if 'color_ramps' in data:
            self.color_ramps = list(data['color_ramps'])

        return super().load(ident, data)

    def open_window(self):
        from .color_picker_window import ColorPickerWindow
        self.colorPickerWindow = ColorPickerWindow(self.color_ramps)
        self.colorPickerWindow.show()
