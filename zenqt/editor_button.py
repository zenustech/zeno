from .editor import *


class QDMGraphicsButton(QGraphicsProxyWidget):
    class QDMLabel(QLabel):
        def __init__(self):
            super().__init__()
            font = QFont()
            font.setPointSize(style['button_text_size'])
            self.setFont(font)

        def mousePressEvent(self, event):
            self.on_click()
            super().mousePressEvent(event)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = self.QDMLabel()
        self.widget.setAlignment(Qt.AlignCenter)
        self.widget.on_click = self.on_click
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


class QDMGraphicsCollapseButton(QGraphicsProxyWidget):
    class QDMCollapseButton(QSvgWidget):
        def __init__(self, parent=None):
            super().__init__()
            self.render = self.renderer()
            self.setStyleSheet('background-color: {}'.format(style['title_color']))

        def mousePressEvent(self, event):
            super().mouseMoveEvent(event)
            self.on_click()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.node = parent

        self.widget = self.QDMCollapseButton(parent)
        self.setWidget(self.widget)
        self.widget.on_click = self.on_click
        self.update_svg()

    def update_svg(self):
        if self.node.collapsed:
            self.widget.load(asset_path('collapse.svg'))
        else:
            self.widget.load(asset_path('unfold.svg'))
        self.widget.render.setAspectRatioMode(Qt.KeepAspectRatio)

    def on_click(self):
        self.setCollapsed(not self.node.collapsed)

    def setCollapsed(self, collapsed):
        self.node.collapsed = collapsed
        if self.node.collapsed:
            self.node.collapse()
        else:
            self.node.unfold()
        self.update_svg()