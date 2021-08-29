from . import *


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
        self.label = QLabel()
        self.label.setFont(font)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def setAlignment(self, align):
        self.edit.setAlignment(align)

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
        if not default: return
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
        text = super().getValue()
        if not text: return None
        return int(text)


class QDMGraphicsParam_bool(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QIntValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        if not default: return
        default = [bool(int(x)) for x in default.split()]
        if len(default) == 1:
            x = default[0]
            self.setValue(x)
        else:
            assert False, default

    def setValue(self, x):
        super().setValue(str(int(x)))

    def getValue(self):
        text = super().getValue()
        if not text: return None
        return bool(int(text))


class QDMGraphicsParam_float(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

        self.validator = QDoubleValidator()
        self.edit.setValidator(self.validator)

    def setDefault(self, default):
        if not default: return
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
        text = super().getValue()
        if not text: return None
        return float(text)


class QDMGraphicsParam_string(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()

    def getValue(self):
        return str(self.edit.text())


class QDMGraphicsParam_writepath(QDMGraphicsParam):
    def initLayout(self):
        super().initLayout()
        self.button = QPushButton('..')
        self.button.setFixedWidth(20)
        self.button.clicked.connect(self.on_open)
        self.layout.addWidget(self.button)

    def getValue(self):
        return str(self.edit.text())

    def on_open(self):
        path, kind = QFileDialog.getSaveFileName(None, 'Path to Save',
            '', 'All Files(*);;')
        if not path:
            return
        self.edit.setText(path)


class QDMGraphicsParam_readpath(QDMGraphicsParam_writepath):
    def on_open(self):
        path, kind = QFileDialog.getOpenFileName(None, 'File to Open',
            '', 'All Files(*);;')
        if not path:
            return
        self.edit.setText(path)


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
        from ...system.utils import os_name
        if os_name == 'win32':  # the stupid win seems doesn't support background-color
            self.edit.setStyleSheet('background-color: white; color: black')
        else:
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
        rect.setHeight(10 * TEXT_HEIGHT)
        super().setGeometry(rect)

    def setValue(self, value):
        self.edit.setPlainText(str(value))

    def getValue(self):
        return str(self.edit.toPlainText())

