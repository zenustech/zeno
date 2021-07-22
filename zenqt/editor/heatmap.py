from . import *



class QDMGraphicsItemNoDragThrough(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)
        super().hoverLeaveEvent(event)


class QDMGraphicsRampDragger(QDMGraphicsItemNoDragThrough):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.parent = parent
        self.selected = False

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
        if self.selected:
            color = style['selected_color']
        else:
            color = style['line_color']
        pen.setColor(QColor(color))
        pen.setWidth(style['ramp_outline_width'])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(-self.width / 2, 0, self.width, self.height)

    def getValue(self):
        f = self.pos().x()
        f = max(0, min(1, f / self.parent.rect.width()))
        return f

    def setValue(self, x):
        self.setX(x * self.parent.rect.width())

    def setX(self, x):
        x = max(0, min(self.parent.rect.width(), x))
        self.setPos(x, 0)

    def incX(self, dx):
        #self.setX(self.pos().x() + dx)
        self.setX(self.pos().x())
        self.parent.updateRamps()

    def setSelected(self, selected):
        super().setSelected(selected)
        self.selected = selected

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self.remove()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if hasattr(self.parent, 'updateRampSelection'):
            self.parent.updateRampSelection(self)
        self.incX(event.pos().x())

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.incX(event.pos().x())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.incX(event.pos().x())

    def remove(self):
        if hasattr(self.parent, 'removeRamp'):
            self.parent.removeRamp(self)


class QDMGraphicsColorChannel(QDMGraphicsItemNoDragThrough):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def setGeometry(self, rect):
        self.setPos(rect.x(), rect.y())
        self.rect = QRectF(rect)
        self.dragger = QDMGraphicsRampDragger(self)

    def setColor(self, r, g, b):
        self.color = (r, g, b)

    def boundingRect(self):
        return QRectF(0, 0, self.rect.width(), self.rect.height())

    def getValue(self):
        return self.dragger.getValue()

    def setValue(self, x):
        return self.dragger.setValue(x)

    def updateRamps(self):
        self.parent.updateRampColor()

    def paint(self, painter, styleOptions, widget=None):
        painter.setPen(Qt.NoPen)
        grad = QLinearGradient(0, 0, self.rect.width(), 0)
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(*self.color))
        brush = QBrush(grad)
        painter.setBrush(brush)
        painter.drawRect(0, 0, self.rect.width(), self.rect.height())


class QDMGraphicsColorRamp(QDMGraphicsItemNoDragThrough):
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

    def updateRampSelection(self, this_dragger):
        for dragger in self.draggers:
            dragger.setSelected(False)
        this_dragger.setSelected(True)
        self.parent.updateRampSelection()

    def currSelectedIndex(self):
        for i, dragger in enumerate(self.draggers):
            if dragger.selected:
                return i
        return None

    def updateRampColor(self, r, g, b):
        i = self.currSelectedIndex()
        if i is not None:
            f, old_rgb = self.ramps[i]
            self.ramps[i] = f, (r, g, b)

    def removeRamp(self, dragger):
        index = self.draggers.index(dragger)
        del self.ramps[index]
        self.initDraggers()

    def addRampAt(self, fac):
        self.ramps.sort(key=lambda x: x[0])
        for i, (lf, lrgb) in reversed(list(enumerate(self.ramps))):
            if fac >= lf:
                break
        else:
            return
        if len(self.ramps) > i + 1:
            rf, rrgb = self.ramps[i + 1]
        else:
            rf, rrgb = lf, lrgb
        intf = (fac - lf) / (rf - lf)
        rgb = tuple((1 - intf) * l + intf * r for l, r in zip(lrgb, rrgb))
        new_ramp = (fac, rgb)
        self.ramps.insert(i, new_ramp)
        self.initDraggers()
        dragger = self.draggers[i]
        self.updateRampSelection(dragger)

    def initDraggers(self):
        for dragger in self.draggers:
            self.scene().removeItem(dragger)
        self.draggers.clear()
        for f, rgb in self.ramps:
            dragger = QDMGraphicsRampDragger(self)
            dragger.setValue(f)
            self.draggers.append(dragger)

    def updateRamps(self):
        for i, dragger in enumerate(self.draggers):
            f = dragger.getValue()
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
        painter.drawRect(0, 0, self.rect.width(), self.rect.height())


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

        self.color_r = QDMGraphicsColorChannel(self)
        self.color_r.setColor(255, 0, 0)
        rect = QRectF(HORI_MARGIN, self.height,
                self.width - 2 * HORI_MARGIN, TEXT_HEIGHT)
        self.color_r.setGeometry(rect)
        self.height += TEXT_HEIGHT * 1.5

        self.color_g = QDMGraphicsColorChannel(self)
        self.color_g.setColor(0, 255, 0)
        rect = QRectF(HORI_MARGIN, self.height,
                self.width - 2 * HORI_MARGIN, TEXT_HEIGHT)
        self.color_g.setGeometry(rect)
        self.height += TEXT_HEIGHT * 1.5

        self.color_b = QDMGraphicsColorChannel(self)
        self.color_b.setColor(0, 0, 255)
        rect = QRectF(HORI_MARGIN, self.height,
                self.width - 2 * HORI_MARGIN, TEXT_HEIGHT)
        self.color_b.setGeometry(rect)
        self.height += TEXT_HEIGHT * 1.5

        self.height += TEXT_HEIGHT

    def updateRampColor(self):
        r = self.color_r.getValue()
        g = self.color_g.getValue()
        b = self.color_b.getValue()
        self.colorramp.updateRampColor(r, g, b)

    def updateRampSelection(self):
        idx = self.colorramp.currSelectedIndex()
        if idx is None: return
        f, (r, g, b) = self.color_ramps[idx]
        self.color_r.setValue(r)
        self.color_g.setValue(g)
        self.color_b.setValue(b)

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
