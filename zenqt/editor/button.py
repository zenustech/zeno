from . import *


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

    def boundingRect(self):
        size = style['collapse_svg_size']
        return QRectF(0, 0, size, size)

    def paint(self, painter, styleOptions, widget=None):
        self.renderer().render(painter, self.boundingRect())


class QDMGraphicsTopButton(QGraphicsSvgItem):
    def __init__(self, parent):
        super().__init__(parent)

        self.node = parent
        self.name = None

        self._renderer = QSvgRenderer(asset_path('unfold.svg'))
        self._renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        self.setSharedRenderer(self._renderer)

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
        self.renderer().render(painter, self.boundingRect())

    def setChecked(self, checked):
        self.checked = checked
        self.update_svg()
        self.update()

    def setWidthHeight(self, width, height):
        self._width = width
        self._height = height

    def on_click(self):
        self.setChecked(not self.checked)

    def mousePressEvent(self, event):
        self.on_click()

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

