from .editor import *


class QDMGraphicsNode_Comment(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.width *= 1.8

    def initSockets(self):
        H = 6 * TEXT_HEIGHT

        self.height += TEXT_HEIGHT * 0.4

        edit = QTextEdit()
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(edit)
        rect = QRectF(HORI_MARGIN, self.height, self.width - HORI_MARGIN * 2, H)
        proxy.setGeometry(rect)

        edit.setText('Comments goes here')
        edit.setStyleSheet('background-color: {}; color: {}'.format(
            style['button_color'], style['button_text_color']))

        self.height += H
        super().initSockets()
