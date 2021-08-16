from . import *


class QDMGraphicsBlackboardResizeHelper(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.setFlag(QGraphicsItem.ItemIsMovable)

        self.node = parent
        self.name = None

        self.setAcceptHoverEvents(True)

    def getCirclePos(self):
        basePos = self.node.pos() + self.pos()
        return basePos

    def getCircleBounds(self):
        return (-SOCKET_RADIUS, -SOCKET_RADIUS,
                2 * SOCKET_RADIUS, 2 * SOCKET_RADIUS)

    def boundingRect(self):
        return QRectF(*self.getCircleBounds()).normalized()

    def paint(self, painter, styleOptions, widget=None):
        painter.setBrush(QColor(style['line_color']))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon([
            QPointF(0, 0),
            QPointF(10, 0),
            QPointF(0, 10),
        ])

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        p = self.pos()
        self.node.setWidthHeight(p.x(), p.y() + TEXT_HEIGHT)

    def hoverEnterEvent(self, event):
        self.node.setFlag(QGraphicsItem.ItemIsMovable, False)
        super().hoverEnterEvent(event)
        self.setCursor(Qt.SizeFDiagCursor)

    def hoverLeaveEvent(self, event):
        self.node.setFlag(QGraphicsItem.ItemIsMovable, True)
        super().hoverLeaveEvent(event)
        self.setCursor(Qt.ArrowCursor)


class QDMGraphicsNode_Blackboard(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setZValue(-2)

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.width = style['node_width']
        self.height = 150

        self.title = QGraphicsTextItem(self)
        self.title.setDefaultTextColor(QColor(style['title_text_color']))
        self.title.setPos(HORI_MARGIN * 2, -TEXT_HEIGHT)
        self.title.setTextInteractionFlags(Qt.TextEditorInteraction)
        font = QFont()
        font.setPointSize(style['title_text_size'])
        self.title.setFont(font)

        self.content = QGraphicsTextItem(self)
        self.content.setDefaultTextColor(QColor(style['title_text_color']))
        self.content.setPos(HORI_MARGIN, HORI_MARGIN)
        self.content.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.content.setFont(font)

        self.helper = QDMGraphicsBlackboardResizeHelper(self)
        self.setWidthHeight(self.width, self.height)

        self.name = None
        self.ident = None

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().moved = True

    def remove(self):
        self.scene().nodes.remove(self)
        self.scene().removeItem(self)

    def setIdent(self, ident):
        self.ident = ident

    def setName(self, name):
        if self.ident is None:
            self.ident = gen_unique_ident(name)
        self.name = name
        self.title.setPlainText(name)

    def initSockets(self):
        pass

    def boundingRect(self):
        return QRectF(0, -TEXT_HEIGHT, self.width, self.height).normalized()

    def paint(self, painter, styleOptions, widget=None):
        r = style['node_rounded_radius']

        pathContent = QPainterPath()
        rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
        pathContent.addRoundedRect(rect, r, r)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['blackboard_panel_color']))
        painter.drawPath(pathContent.simplified())

        # title round top
        pathTitle = QPainterPath()
        rect = QRectF(0, -TEXT_HEIGHT, self.width, TEXT_HEIGHT)
        pathTitle.addRoundedRect(rect, r, r)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['blackboard_title_color']))
        painter.drawPath(pathTitle.simplified())
        
        # title direct bottom
        pathTitle = QPainterPath()
        rect = QRectF(0, -r, self.width, r)
        pathTitle.addRect(rect)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style['blackboard_title_color']))
        painter.drawPath(pathTitle.simplified())

        if self.isSelected():
            pathOutline = QPainterPath()
            rect = QRectF(0, -TEXT_HEIGHT, self.width, self.height)
            pathOutline.addRoundedRect(rect, r, r)
            pen = QPen(QColor(style['selected_color']))
            pen.setWidth(style['node_outline_width'])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(pathOutline.simplified())

    def setWidthHeight(self, width, height):
        width = max(width, style['node_width'])
        height = max(height, 150)
        self.width = width
        self.height = height
        self.helper.setPos(width, height - TEXT_HEIGHT)

        rect = QRectF(HORI_MARGIN, HORI_MARGIN, self.width - HORI_MARGIN * 2,
            self.height - TEXT_HEIGHT - HORI_MARGIN * 2)
        self.content.setTextWidth(self.width - HORI_MARGIN * 2)

    def dump(self):
        uipos = self.pos().x(), self.pos().y()
        data = {
            'name': self.name,
            'uipos': uipos,
            'special': True,
            'width': self.width,
            'height': self.height,
            'title': self.title.toPlainText(),
            'content': self.content.toPlainText(),
            'params': [] # TODO: deprecate
        }
        return self.ident, data
    
    def load(self, ident, data):
        name = data['name']
        posx, posy = data['uipos']

        self.initSockets()
        self.setIdent(ident)
        self.setName(name)
        self.setPos(posx, posy)
        self.setWidthHeight(data['width'], data['height'])

        self.title.setPlainText(data['title'])
        self.content.setPlainText(data['content'])

        edges = []
        return edges

