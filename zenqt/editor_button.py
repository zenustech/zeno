from .editor import *


class QDMGraphicsButton(QGraphicsProxyWidget):
    class QDMSVGButton(QSvgWidget):
        def __init__(self, parent=None):
            super().__init__()
            self.render = self.renderer()
            self.setStyleSheet('background-color: #376557')

        def mousePressEvent(self, event):
            self.on_click()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = self.QDMSVGButton()
        self.widget.on_click = self.on_click
        self.setWidget(self.widget)
        self.checked = False

    def on_click(self):
        self.setChecked(not self.checked)

    def setChecked(self, checked):
        self.checked = checked
        self.update_svg()

    def setText(self, text):
        self.svg_active_path = 'node-button/' + text + '-active.svg'
        self.svg_quiet_path = 'node-button/' + text + '-quiet.svg'
        self.update_svg()

    def update_svg(self):
        if self.checked:
            self.widget.load(asset_path(self.svg_active_path))
        else:
            self.widget.load(asset_path(self.svg_quiet_path))

        self.widget.render.setViewBox(QRectF(-5, -5, 34, 34))
        self.widget.render.setAspectRatioMode(Qt.KeepAspectRatio)


class QDMGraphicsCollapseButton(QGraphicsProxyWidget):
    class QDMCollapseButton(QSvgWidget):
        def __init__(self, parent=None):
            super().__init__()
            self.render = self.renderer()
            self.setStyleSheet('background-color: transparent')

        def mousePressEvent(self, event):
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