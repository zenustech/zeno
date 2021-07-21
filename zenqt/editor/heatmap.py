from . import *


class QDMGraphicsColorRamp(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rect = QRectF()
        self.ramps = [
                (0.0, (0, 0, 0)),
                (0.5, (1, 0, 0)),
                (0.9, (1, 1, 0)),
                (1.0, (1, 1, 1)),
        ]

    def setGeometry(self, rect):
        self.setPos(rect.x(), rect.y())
        self.rect = QRectF(rect)

    def boundingRect(self):
        return QRectF(0, 0, self.rect.width(), self.rect.height())

    def paint(self, painter, styleOptions, widget=None):
        painter.setPen(Qt.NoPen)
        grad = QLinearGradient(0, 0, self.rect.width(), 0)
        for f, (r, g, b) in self.ramps:
            grad.setColorAt(f, QColor(int(r * 255), int(g * 255), int(b * 255)))
        brush = QBrush(grad)
        painter.setBrush(brush)
        painter.drawRect(0, 0, self.rect.width(), TEXT_HEIGHT)


class QDMGraphicsNode_MakeHeatmap(QDMGraphicsNode):
    def __init__(self, parent=None):
        self.color_ramps = []
        super().__init__(parent)

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.7

        self.colorramp = QDMGraphicsColorRamp(self)
        rect = QRectF(HORI_MARGIN, self.height,
                self.width - 2 * HORI_MARGIN, TEXT_HEIGHT)
        self.colorramp.setGeometry(rect)
        self.height += TEXT_HEIGHT * 1.5

        if not hasattr(self, 'add_button'):
            self.add_button = QDMGraphicsButton(self)
        W = self.width / 4
        rect = QRectF(HORI_MARGIN, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.add_button.setGeometry(rect)
        self.add_button.setText('+')
        self.add_button.on_click = self.on_add

        if not hasattr(self, 'del_button'):
            self.del_button = QDMGraphicsButton(self)
        rect = QRectF(HORI_MARGIN + W, self.height,
            W - HORI_MARGIN, TEXT_HEIGHT)
        self.del_button.setGeometry(rect)
        self.del_button.setText('-')
        self.del_button.on_click = self.on_del
        self.height += TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 1.5

    def on_add(self):
        pass

    def on_del(self):
        pass

    def dump(self):
        ident, data = super().dump()
        data['color_ramps'] = tuple(self.color_ramps)
        data['params']['_RAMPS'] = '\n'.join(
                f'{r} {g} {b}' for f, (r, g, b) in self.color_ramps)
        return ident, data

    def load(self, ident, data):
        if 'color_ramps' in data:
            self.color_ramps = list(data['color_ramps'])

        return super().load(ident, data)
