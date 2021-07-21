from . import *


class QDMGraphicsColorRamp(QGraphicsItem):
    class QDMGraphicsRampDragger(QGraphicsItem):
        def __init__(self, parent):
            super().__init__(parent)
            self.setFlag(QGraphicsItem.ItemIsSelectable)
            self.parent = parent

        @property
        def width(self):
            return style['ramp_width']

        @property
        def height(self):
            return self.parent.rect.height()

        def boundingRect(self):
            return QRectF(-self.width / 2, 0, self.width, self.height)

        def paint(self, painter, styleOptions, widget=None):
            pen = QPen()
            color = style['selected_color'] if self.isSelected() else style['line_color']
            pen.setColor(QColor(color))
            pen.setWidth(style['ramp_outline_width'])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(-self.width / 2, 0, self.width, self.height)

        def setX(self, x):
            x = max(0, min(self.parent.rect.width(), x))
            self.setPos(x, 0)

        def incX(self, dx):
            self.setX(self.pos().x() + dx)
            self.parent.updateRamps()

        def mousePressEvent(self, event):
            super().mousePressEvent(event)
            self.incX(event.pos().x())

        def mouseMoveEvent(self, event):
            super().mouseMoveEvent(event)
            self.incX(event.pos().x())

        def mouseReleaseEvent(self, event):
            super().mouseReleaseEvent(event)
            self.incX(event.pos().x())

        def remove(self):
            self.parent.removeRamp(self)

    def __init__(self, parent):
        super().__init__(parent)
        self.rect = QRectF()
        self.parent = parent

        self.draggers = []

    def mousePressEvent(self, event):
        f = event.pos().x()
        if 0 <= f <= self.rect.width():
            f /= self.rect.width()
            self.addRampAt(f)

    def sortRamps(self):
        print(self.ramps)
        self.ramps.sort(key=lambda x: x[0])
        print(self.ramps)

    def removeRamp(self, dragger):
        index = self.draggers.index(dragger)
        del self.ramps[index]
        self.initDraggers()

    def addRampAt(self, fac):
        print('add at', fac)
        self.sortRamps()
        for i, (oldf, rgb) in enumerate(list(self.ramps)):
            if fac >= oldf:
                rgb = (0, 0, 0)
                new_ramp = (fac, rgb)
                self.ramps.insert(i, new_ramp)
                break
        self.initDraggers()

    def initDraggers(self):
        for dragger in self.draggers:
            self.scene().removeItem(dragger)
        self.draggers.clear()
        for f, rgb in self.ramps:
            dragger = self.QDMGraphicsRampDragger(self)
            dragger.setX(f * self.rect.width())
            self.draggers.append(dragger)

    def updateRamps(self):
        for i, dragger in enumerate(self.draggers):
            f = dragger.pos().x()
            f = max(0, min(1, f / self.rect.width()))
            _, rgb = self.ramps[i]
            self.ramps[i] = f, rgb
        self.update()

    @property
    def ramps(self):
        return self.parent.color_ramps

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
        self.color_ramps = [
                (0.0, (0, 0, 0)),
                (0.5, (1, 0, 0)),
                (0.9, (1, 1, 0)),
                (1.0, (1, 1, 1)),
        ]
        super().__init__(parent)

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.7

        self.colorramp = QDMGraphicsColorRamp(self)
        rect = QRectF(HORI_MARGIN, self.height,
                self.width - 2 * HORI_MARGIN, TEXT_HEIGHT)
        self.colorramp.setGeometry(rect)
        self.colorramp.initDraggers()
        self.height += TEXT_HEIGHT * 1.5

        self.height += TEXT_HEIGHT * 1.5

    def dump(self):
        ident, data = super().dump()
        data['color_ramps'] = tuple(self.color_ramps)
        data['params']['_RAMPS'] = '\n'.join(
                f'{f} {r} {g} {b}' for f, (r, g, b) in self.color_ramps)
        return ident, data

    def load(self, ident, data):
        if 'color_ramps' in data:
            self.color_ramps = list(data['color_ramps'])

        return super().load(ident, data)
