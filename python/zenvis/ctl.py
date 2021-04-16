from . import core

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *



class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.value_changed)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)

        self.label = QLabel('-')

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)


    def value_changed(self):
        self.frameid = self.slider.value()
        self.label.setText(str(self.frameid))

    @property
    def solver_frameid(self):
        return core.get_solver_frameid()

    @property
    def frameid(self):
        return core.get_curr_frameid()

    @frameid.setter
    def frameid(self, value):
        core.set_curr_frameid(value)

