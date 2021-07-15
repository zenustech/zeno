from .editor import *

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
        self.dummy = False

        self.offset = 12
        self.text_offset = HORI_MARGIN * 2 - 5
        self.label.setPos(self.text_offset, - style['socket_text_size'] * 1.2)

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
            return (self.node.width - SOCKET_RADIUS - self.offset, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)
        else:
            return (-SOCKET_RADIUS + self.offset, -SOCKET_RADIUS,
                    2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        if self.hasAnyEdge() or self.dummy:
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

