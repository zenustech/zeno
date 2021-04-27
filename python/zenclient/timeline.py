from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from zenwebcfg import zenvis


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

    def on_update(self):
        frameid = zenvis.dnStat['frameid']
        self.slider.setValue(frameid)
        self.label.setText(str(frameid))
        self.status.setText(self.get_status_string())

    def value_changed(self):
        zenvis.upStat['next_frameid'] = self.slider.value()
        zenvis.upStat['playing'] = self.player.isChecked()

    def get_status_string(self):
        fps = zenvis.dnStat['render_fps']
        spf = zenvis.dnStat['solver_interval']
        return f'{fps:.1f} FPS | {spf:.02f} secs/step'
