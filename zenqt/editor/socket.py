from . import *


class QDMGraphicsSocket(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.isOutput = False
        self.edges = set()

        self.node = parent
        self.name = None

        self.setAcceptHoverEvents(True)
        self.hovered = False
        self.temp_edge_connected = False

        self.initLabel()

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    class QDMGraphicsTextItem(QGraphicsTextItem):
        def __init__(self, parent):
            super().__init__(parent)
            self.setDefaultTextColor(QColor(style['socket_text_color']))
            self.parent = parent

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
            rect = self.label.boundingRect()
            x = self.node.boundingRect().width() - rect.width() - HORI_MARGIN
            self.label.setPos(x, self.label.pos().y())
            #self.label.setAlignment(Qt.AlignRight)
            #if hasattr(self.label, 'setTextWidth'):
                #width = self.node.boundingRect().width() - HORI_MARGIN * 2
                #self.label.setTextWidth(width)

    def setName(self, name):
        self.name = name
        self.label.setPlainText(name)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        if not self.isOutput:
            return basePos + QPointF(style['socket_offset'], 0)
        else:
            return basePos + QPointF(self.node.width - style['socket_offset'], 0)

    def getCircleBounds(self):
        if not self.isOutput:
            return (-SOCKET_RADIUS + style['socket_offset'], -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)
        else:
            return (self.node.width - SOCKET_RADIUS - style['socket_offset'], -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        if self.hasAnyEdge() or self.hovered or self.temp_edge_connected:
            socket_color = 'socket_connect_color'
        else:
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
        offset = style['dummy_socket_offset']
        if not self.isOutput:
            return basePos + QPointF(-offset, 0)
        else:
            return basePos + QPointF(self.node.width + offset, 0)

    def getCircleBounds(self):
        h = style['dummy_socket_height']
        w = style['dummy_socket_width']
        offset = style['dummy_socket_offset'] // 2
        if not self.isOutput:
            return QRectF(-w -offset, - (h // 2), w, h)
        else:
            return QRectF(self.node.width + offset, - (h // 2), w, h)

    def boundingRect(self):
        return self.getCircleBounds().normalized()

    def paint(self, painter, styleOptions, widget=None):
        rect = self.getCircleBounds()
        fillRect(painter, rect, style['dummy_socket_color'])


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
        button_color = style['top_button_color']
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

        s = style['top_svg_size']
        p = style['top_svg_padding']

        self._renderer.setViewBox(QRectF(-p, -p, s + p * 2, s + p * 2))
        self._renderer.setAspectRatioMode(Qt.KeepAspectRatio)

