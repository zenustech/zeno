from . import *


class QDMGraphicsSocket(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.isOutput = False
        self.edges = set()

        self.node = parent
        self.name = None
        self.dummy = False

        self.initLabel()

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
        self.label.setPlainText(translate(name))

    def setType(self, type):
        self.type = type
        #self.label.setPlainText(self.name + ' (' + type + ')')

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
        if hasattr(self, 'paramEdit'):
            if self.hasAnyEdge() or self.dummy:
                self.paramEdit.hide()
            else:
                self.paramEdit.show()

        if self.hasAnyEdge() or self.dummy:
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

    def setDefault(self, default):
        if self.isOutput:
            return
        if not self.type:
            return
        type = self.type
        if type == 'NumericObject':
            type = 'float'  # for convinent editing for NumericOperator..
        if type.startswith('enum '):
            self.paramEdit = QDMGraphicsParamEnum(self)
            enums = type.split()[1:]
            self.paramEdit.setEnums(enums)
        else:
            param_type = 'QDMGraphicsParam_' + type
            if param_type not in globals():
                return
            self.paramEdit = globals()[param_type](self)
        w = self.label.boundingRect().width()
        rect = QRectF(HORI_MARGIN + w, -TEXT_HEIGHT * 0.5,
            self.node.width - HORI_MARGIN * 3 - w, 0)
        self.paramEdit.setGeometry(rect)
        self.paramEdit.setDefault(default)

    def setValue(self, value):
        if hasattr(self, 'paramEdit'):
            self.paramEdit.setValue(value)

    def getValue(self):
        if hasattr(self, 'paramEdit'):
            return self.paramEdit.getValue()
        return None
