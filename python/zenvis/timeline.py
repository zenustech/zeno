from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .coredll import core


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QLabel('0')

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.value_changed)
        self.slider.setMinimum(0)
        self.slider.setMaximum(250)

        self.player = QCheckBox('Play')
        self.player.setChecked(True)

        layout = QHBoxLayout()
        layout.addWidget(self.player)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.startTimer(0)

    def timerEvent(self, event):
        if self.player.isChecked():
            self.frameid += 1
        else:
            self.frameid = self.frameid

    def value_changed(self):
        self.frameid = self.slider.value()

    @property
    def solver_frameid(self):
        return core.get_solver_frameid()

    @property
    def frameid(self):
        return core.get_curr_frameid()

    @frameid.setter
    def frameid(self, value):
        core.set_curr_frameid(value)
        value = core.get_curr_frameid()
        self.label.setText(str(value))
        self.slider.setValue(value)
