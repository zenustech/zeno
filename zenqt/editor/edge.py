from . import *


class QDMGraphicsPath(QGraphicsPathItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(-1)

        self.srcPos = QPointF(0, 0)
        self.dstPos = QPointF(0, 0)

    def setSrcDstPos(self, srcPos, dstPos):
        self.srcPos = srcPos
        self.dstPos = dstPos

    def paint(self, painter, styleOptions, widget=None):
        self.updatePath()
        color = 'selected_color' if self.isSelected() else 'line_color'
        pen = QPen(QColor(style[color]))
        pen.setWidth(style['line_width'])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    '''
    def boundingRect(self):
        x0 = min(self.srcPos.x(), self.dstPos.x())
        y0 = min(self.srcPos.y(), self.dstPos.y())
        x1 = max(self.srcPos.x(), self.dstPos.x())
        y1 = max(self.srcPos.y(), self.dstPos.y())
        return QRectF(x0, y0, x1 - x0, y1 - y0)
    '''

    def updatePath(self):
        path = QPainterPath(self.srcPos)
        if BEZIER_FACTOR == 0:
            path.lineTo(self.dstPos.x(), self.dstPos.y())
        else:
            dist = self.dstPos.x() - self.srcPos.x()
            dist = max(100, dist, -dist) * BEZIER_FACTOR
            path.cubicTo(self.srcPos.x() + dist, self.srcPos.y(),
                    self.dstPos.x() - dist, self.dstPos.y(),
                    self.dstPos.x(), self.dstPos.y())
        self.setPath(path)


class QDMGraphicsTempEdge(QDMGraphicsPath):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.item = None
        self.endPos = None

    def setItem(self, item):
        self.item = item

    def setEndPos(self, pos):
        self.endPos = pos

    def updatePath(self):
        if self.item.isOutput:
            self.setSrcDstPos(self.item.getCirclePos(), self.endPos)
        else:
            self.setSrcDstPos(self.endPos, self.item.getCirclePos())

        super().updatePath()


class QDMGraphicsEdge(QDMGraphicsPath):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.srcSocket = None
        self.dstSocket = None

    def setSrcSocket(self, socket):
        assert socket.isOutput
        socket.addEdge(self)
        self.srcSocket = socket

    def setDstSocket(self, socket):
        assert not socket.isOutput
        socket.addEdge(self)
        self.dstSocket = socket

    def updatePath(self):
        s = self.srcSocket
        if not s.node.collapsed:
            srcPos = s.getCirclePos()
        else:
            srcPos = s.node.dummy_output_socket.getCirclePos()
        s = self.dstSocket
        if not s.node.collapsed:
            dstPos = s.getCirclePos()
        else:
            dstPos = s.node.dummy_input_socket.getCirclePos()
        self.setSrcDstPos(srcPos, dstPos)

        super().updatePath()

    def remove(self):
        if self.srcSocket is not None:
            if self in self.srcSocket.edges:
                self.srcSocket.edges.remove(self)
        if self.dstSocket is not None:
            if self in self.dstSocket.edges:
                self.dstSocket.edges.remove(self)

        self.scene().removeItem(self)

