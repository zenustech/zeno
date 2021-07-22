from . import *


class QDMGraphicsSocket(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.isOutput = False
        self.edges = set()

        self.node = parent
        self.name = None
        self.dummy = False

        self.setAcceptHoverEvents(True)
        self.hovered = False

        self.initLabel()

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    class QDMGraphicsTextItem(QGraphicsTextItem):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setDefaultTextColor(QColor(style['socket_text_color']))

        def setAlignment(self, align):
            document = self.document()
            option = document.defaultTextOption()
            option.setAlignment(Qt.AlignRight)
            document.setDefaultTextOption(option)

    def initLabel(self):
        self.label = self.QDMGraphicsTextItem(self)
        self.label.setPos(HORI_MARGIN, -TEXT_HEIGHT * 0.5)
        font = QFont()
        font.setPointSize(style['socket_text_size'])
        self.label.setFont(font)

    def hasAnyEdge(self):
        return len(self.edges) != 0

    def getTheOnlyEdge(self):
        assert not self.isOutput
        assert len(self.edges) == 1
        return next(iter(self.edges))

    def removeAllEdges(self):
        for edge in list(self.edges):
            edge.remove()
        assert len(self.edges) == 0

    def addEdge(self, edge):
        if not self.isOutput:
            self.removeAllEdges()
        self.edges.add(edge)

    def setIsOutput(self, isOutput):
        self.isOutput = isOutput

        if isOutput:
            self.label.setAlignment(Qt.AlignRight)
            if hasattr(self.label, 'setTextWidth'):
                width = self.node.boundingRect().width() - HORI_MARGIN * 2
                self.label.setTextWidth(width)

    def setName(self, name):
        self.name = name
        self.label.setPlainText(name)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        if self.isOutput:
            return basePos + QPointF(self.node.width, 0)
        else:
            return basePos

    def getCircleBounds(self):
        if self.isOutput:
            return (self.node.width - SOCKET_RADIUS, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)
        else:
            return (-SOCKET_RADIUS, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        if self.hasAnyEdge() or self.hovered:
            socket_color = 'socket_connect_color'
        else:
            socket_color = 'socket_unconnect_color'
        painter.setBrush(QColor(style[socket_color]))
        pen = QPen(QColor(style['line_color']))
        pen.setWidth(style['socket_outline_width'])
        painter.setPen(pen)
        painter.drawEllipse(*self.getCircleBounds())

    def remove(self):
        for edge in list(self.edges):
            edge.remove()


class QDMGraphicsButton(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.node = parent
        self.name = None

        self.initLabel()
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
        button_color = 'button_selected_color' if self.checked else 'button_color'
        painter.fillRect(*self.getCircleBounds(), QColor(style[button_color]))

    def setChecked(self, checked):
        self.checked = checked

        text_color = 'button_selected_text_color' if self.checked else 'button_text_color'
        self.label.setDefaultTextColor(QColor(style[text_color]))

    def setWidthHeight(self, width, height):
        self._width = width
        self._height = height
        self.label.setTextWidth(width)

    def on_click(self):
        self.setChecked(not self.checked)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.on_click()

    def setGeometry(self, rect):
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        self.setPos(x, y)
        self.setWidthHeight(w, h)


class QDMGraphicsCollapseButton(QGraphicsSvgItem):
    def __init__(self, parent):
        super().__init__(parent)
        self.node = parent

        self._renderer = QSvgRenderer(asset_path('unfold.svg'))
        self.update_svg(False)

    def update_svg(self, collapsed):
        svg_filename = ('collapse' if collapsed else 'unfold') + '.svg'
        self._renderer.load(asset_path(svg_filename))
        self._renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        self.setSharedRenderer(self._renderer)

    def mousePressEvent(self, event):
        super().mouseMoveEvent(event)
        self.node.collapsed = not self.node.collapsed
        if self.node.collapsed:
            self.node.collapse()
        else:
            self.node.unfold()

