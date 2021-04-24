from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .coredll import core


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.label = QLabel('-')
        self.status = QLabel('-')

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
        layout.addWidget(self.status)
        self.setLayout(layout)

        self.startTimer(1000 // 60)

    def timerEvent(self, event):
        self.slider.setValue(self.frameid)
        self.label.setText(str(self.frameid))
        self.status.setText(self.get_status_string())
        if self.player.isChecked():
            self.frameid += 1

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

    def get_status_string(self):
        fps = core.get_render_fps()
        spf = core.get_solver_interval()
        stat = f'{fps:.1f} FPS | {spf:.02f} secs/step'
        return stat