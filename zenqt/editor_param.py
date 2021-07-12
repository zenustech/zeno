from .editor import *

class QDMGraphicsParam(QGraphicsProxyWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.initLayout()
        if hasattr(self.edit, 'editingFinished'):
            self.edit.editingFinished.connect(self.edit_finished)
        assert hasattr(self, 'layout')

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.widget.setStyleSheet('background-color: {}; color: #eeeeee'.format(style['panel_color']))

        self.setWidget(self.widget)
        self.setContentsMargins(0, 0, 0, 0)

        self.name = None

    def edit_finished(self):
        self.parent.scene().record()

    def initLayout(self):
        font = QFont()
        font.setPointSize(style['param_text_size'])

        self.edit = QLineEdit()
        self.edit.setFont(font)
        self.edit.setStyleSheet('background-color: #252525; color: #787878; border-style:outset')
        self.label = QLabel()
        self.label.setFont(font)
        self.label.setStyleSheet('color: #787878')

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def setName(self, name):
        self.name = name
        self.label.setText(name)

    def setDefault(self, default):
        self.setValue(default)

    def getValue(self):
        return str(self.edit.text())

    def setValue(self, value):
        self.edit.setText(str(value))


class QDMGraphicsParam_int(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QIntValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        default = [int(x) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.setValue(x)
        elif len(default) == 2:
            x, xmin = default
            self.setValue(x)
            self.validator.setBottom(xmin)
        elif len(default) == 3:
            x, xmin, xmax = default
            self.setValue(x)
            self.validator.setBottom(xmin)
            self.validator.setTop(xmax)
        else:
            assert False, default

    def getValue(self):
        return int(self.edit.text())


class QDMGraphicsParam_float(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QDoubleValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        default = [float(x) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.setValue(x)
        elif len(default) == 2:
            x, xmin = default
            self.setValue(x)
            self.validator.setBottom(xmin)
        elif len(default) == 3:
            x, xmin, xmax = default
            self.setValue(x)
            self.validator.setBottom(xmin)
            self.validator.setTop(xmax)
        else:
            assert False, default

    def getValue(self):
        return float(self.edit.text())



class QDMGraphicsParam_string(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

    def getValue(self):
        return str(self.edit.text())


class QDMGraphicsParam_multiline_string(QDMGraphicsParam):
    class QDMPlainTextEdit(QPlainTextEdit):
        def focusOutEvent(self, event):
            self.parent.edit_finished()
            super().focusOutEvent(event)

    def initLayout(self):
        font = QFont()
        font.setPointSize(style['param_text_size'])

        self.edit = self.QDMPlainTextEdit()
        self.edit.parent = self
        self.edit.setFont(font)
        self.edit.setStyleSheet('background-color: {}; color: {}'.format(
            style['button_color'], style['button_text_color']))

        self.label = QLabel()
        self.label.setFont(font)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setWidget(self.edit)

    def setGeometry(self, rect):
        rect = QRectF(rect)
        rect.setHeight(6 * TEXT_HEIGHT)
        super().setGeometry(rect)

    def setValue(self, value):
        self.edit.setPlainText(str(value))

    def getValue(self):
        return str(self.edit.toPlainText())


'''
class QDMGraphicsNode_MakeMultilineString(QDMGraphicsNode):
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

        edit.setText('')
        edit.setStyleSheet('background-color: {}; color: {}'.format(
            style['button_color'], style['button_text_color']))
        font = QFont()
        font.setPointSize(style['param_text_size'])
        edit.setFont(font)

        self.height += H
        super().initSockets()
'''
