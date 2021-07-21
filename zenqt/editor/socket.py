from . import *


class QDMGraphicsSocket(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.label = QGraphicsTextItem(self)
        self.label.setDefaultTextColor(QColor(style['socket_text_color']))
        font = QFont()
        font.setPointSize(style['socket_text_size'])
        font.setWeight(QFont.DemiBold)
        self.label.setFont(font)

        self.isOutput = False
        self.edges = set()

        self.node = parent
        self.name = None

        self.offset = 13
        self.text_offset = HORI_MARGIN * 2 - 2
        self.label.setPos(self.text_offset, - style['socket_text_size'] * 1.3)

        self._hover = False
        self.setAcceptHoverEvents(True)

    def hoverMoveEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()

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
            document = self.label.document()
            option = document.defaultTextOption()
            option.setAlignment(Qt.AlignRight)
            document.setDefaultTextOption(option)
            width = self.node.boundingRect().width() - self.text_offset * 2
            self.label.setTextWidth(width)

    def setName(self, name):
        self.name = name
        self.label.setPlainText(name)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        if self.isOutput:
            return basePos + QPointF(self.node.width, 0) + QPointF(-self.offset, 0)
        else:
            return basePos + QPointF(self.offset, 0)

    def getCircleBounds(self):
        SOCKET_RADIUS = 3
        if self.isOutput:
            return (self.node.width - SOCKET_RADIUS - self.offset - 1, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)
        else:
            return (-SOCKET_RADIUS + self.offset, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        b = self.getCircleBounds()
        offset = 2
        return QRectF(b[0] - offset, b[1] - offset,
                b[2] + offset * 2, b[3] + offset * 2).normalized()

    def paint(self, painter, styleOptions, widget=None):
        if self.hasAnyEdge() or self._hover:
            self.label.setDefaultTextColor(QColor(style['socket_connect_color']))
            socket_color = 'socket_connect_color'
        else:
            self.label.setDefaultTextColor(QColor(style['socket_text_color']))
            socket_color = 'socket_unconnect_color'
        painter.setBrush(QColor(style[socket_color]))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(*self.getCircleBounds())

    def remove(self):
        for edge in list(self.edges):
            edge.remove()


class QDMGraphicsDummySocket(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.isOutput = False
        self.node = parent

    def setIsOutput(self, isOutput):
        self.isOutput = isOutput

    def getCirclePos(self):
        basePos = self.node.pos()
        offset = 3
        if not self.isOutput:
            return basePos + QPointF(-offset, 0)
        else:
            return basePos + QPointF(self.node.width + offset, 0)

    def getCircleBounds(self):
        h = 40
        w = 5
        offset = 1
        if not self.isOutput:
            return QRectF(-w -offset, - (h // 2), w, h)
        else:
            return QRectF(self.node.width + offset, - (h // 2), w, h)

    def boundingRect(self):
        return self.getCircleBounds().normalized()

    def paint(self, painter, styleOptions, widget=None):
        rect = self.getCircleBounds()
        fillRect(painter, rect, '#4D4D4D')


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
        self.on_click()

    def setGeometry(self, rect):
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        self.setPos(x, y)
        self.setWidthHeight(w, h)


class QDMGraphicsTopButton(QGraphicsSvgItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.node = parent
        self.name = None

        self._renderer = QSvgRenderer(asset_path('unfold.svg'))
        self._renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        self.setSharedRenderer(self._renderer)

        self.setChecked(False)

    def setText(self, name):
        self.name = name
        self.svg_active_path = 'node-button/' + name + '-active.svg'
        self.svg_mute_path = 'node-button/' + name + '-mute.svg'

    def getCircleBounds(self):
        return (0, 0, self._width, self._height)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        button_color = '#376557'
        painter.fillRect(*self.getCircleBounds(), QColor(button_color))
        self.update_svg()
        self.renderer().render(painter, self.boundingRect())

    def setChecked(self, checked):
        self.checked = checked

    def setWidthHeight(self, width, height):
        self._width = width
        self._height = height

    def on_click(self):
        self.setChecked(not self.checked)

    def mousePressEvent(self, event):
        self.on_click()
        self.update()

    def setGeometry(self, rect):
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        self.setPos(x, y)
        self.setWidthHeight(w, h)

    def update_svg(self):
        if self.checked:
            self._renderer.load(asset_path(self.svg_active_path))
        else:
            self._renderer.load(asset_path(self.svg_mute_path))

        self._renderer.setViewBox(QRectF(-5, -5, 34, 34))
        self._renderer.setAspectRatioMode(Qt.KeepAspectRatio)


class QDMGraphicsCollapseButton(QGraphicsSvgItem):
    def __init__(self, parent):
        super().__init__(parent)
        self.node = parent

        self.width = 28
        self.height = 28

        self._renderer = QSvgRenderer(asset_path('unfold.svg'))
        self.update_svg(False)

    def update_svg(self, collapsed):
        svg_filename = ('collapse' if collapsed else 'unfold') + '.svg'
        self._renderer.load(asset_path(svg_filename))
        self._renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        self.setSharedRenderer(self._renderer)

    def mousePressEvent(self, event):
        self.node.collapsed = not self.node.collapsed
        if self.node.collapsed:
            self.node.collapse()
        else:
            self.node.unfold()
    
    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter, styleOptions, widget=None):
        self.renderer().render(painter, self.boundingRect())

