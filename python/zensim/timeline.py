from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import zenvis


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
        self.player.clicked.connect(self.value_changed)
        self.player.setChecked(True)

        layout = QHBoxLayout()
        layout.addWidget(self.player)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.status)
        self.setLayout(layout)

        self.startTimer(1000 // 60)

    def timerEvent(self, event):
        frameid = zenvis.sendBuf['frameid']
        self.slider.setValue(frameid)
        self.label.setText(str(frameid))
        self.status.setText(self.get_status_string())

    def value_changed(self):
        zenvis.sendBuf['frameid'] = self.slider.value()
        zenvis.sendBuf['playing'] = self.player.isChecked()

    def get_status_string(self):
        fps = zenvis.sendBuf['render_fps']
        spf = zenvis.sendBuf['solver_interval']
        return f'{fps:.1f} FPS | {spf:.02f} secs/step'
