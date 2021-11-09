from . import *

class QDMOneClickButton(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.node = parent
        self.name = None

        self.initLabel()
        self.checked = False
        self.setChecked(False)

    def initLabel(self):
        self.label = QGraphicsTextItem(self)
        self.label.setPos(0, - TEXT_HEIGHT * 0.1)
        font = QFont()
        font.setPointSize(style['socket_text_size'])
        self.label.setFont(font)

        document = self.label.document()
        option = document.defaultTextOption()
        option.setAlignment(Qt.AlignCenter)
        document.setDefaultTextOption(option)

    def setText(self, name):
        self.name = name
        self.label.setPlainText(name)

    def getCircleBounds(self):
        return (0, 0, self._width, self._height)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        button_color = 'button_color'
        painter.fillRect(*self.getCircleBounds(), QColor(style[button_color]))

    def setChecked(self, checked):
        self.checked = checked

        text_color = 'button_text_color'
        self.label.setDefaultTextColor(QColor(style[text_color]))

    def setWidthHeight(self, width, height):
        self._width = width
        self._height = height
        self.label.setTextWidth(width)

    def on_click(self):
        self.setChecked(not self.checked)
        from .curve_editor import CurveEditor
        self.curve_editor = CurveEditor(self.node)
        self.curve_editor.open()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.on_click()

    def setGeometry(self, rect):
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        self.setPos(x, y)
        self.setWidthHeight(w, h)

class QDMGraphicsNode_MakeCurveMap(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = [
            (0, 0),
            (1, 1),
        ]
        self.handlers = [
            [(0, 0), (1/6, 1/6)],
            [(-1/6, -1/6), (0, 0)],
        ]

    def initSockets(self):
        super().initSockets()
        self.height -= TEXT_HEIGHT * 0.7

        button = QDMOneClickButton(self)
        button.setPos(50, self.height)
        button.setWidthHeight(100, 20)
        button.setText('Edit')
        self.height += 60

    def dump(self):
        ident, data = super().dump()
        points = tuple(sorted(self.points, key=lambda x: x[0]))
        data['points'] = tuple(points)
        return ident, data

    def load(self, ident, data):
        self.points = list(data['points'])

        return super().load(ident, data)
