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


class QDMCollapseButton(QSvgWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.render = self.renderer()
        self.load(asset_path('unfold.svg'))
        # PySide2 >= 5.15
        self.render.setAspectRatioMode(Qt.KeepAspectRatio)

        self.setStyleSheet('background-color: {}'.format(style['title_color']))
        self.node = parent
    
    def isChecked(self):
        return self.collapseds
    
    def mousePressEvent(self, event):
        super().mouseMoveEvent(event)
        self.node.collapsed = not self.node.collapsed
        if self.node.collapsed:
            self.node.collapse()
        else:
            self.node.unfold()

    def update_svg(self):
        if self.node.collapsed:
            self.load(asset_path('collapse.svg'))
        else:
            self.load(asset_path('unfold.svg'))
        self.render.setAspectRatioMode(Qt.KeepAspectRatio)


class QDMGraphicsCollapseButton(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = QDMCollapseButton(parent)
        self.setWidget(self.widget)

    def update_svg(self):
        self.widget.update_svg()


