from .editor import *


class QDMGraphicsNode_Subgraph(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)  # todo: support dyn-node-desc


class QDMGraphicsTextEdit(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = QPushButton()
        self.widget.clicked.connect(self.on_click)
        self.setWidget(self.widget)
        self.setChecked(False)

    def on_click(self):
        self.setChecked(not self.checked)

    def setChecked(self, checked):
        self.checked = checked
        if self.checked:
            self.widget.setStyleSheet('background-color: {}; color: {}'.format(
                style['button_selected_color'], style['button_selected_text_color']))
        else:
            self.widget.setStyleSheet('background-color: {}; color: {}'.format(
                style['button_color'], style['button_text_color']))

    def setText(self, text):
        self.widget.setText(text)


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
