from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtSvg import *

from ..utils import asset_path, setKeepAspect
from . import zenvis

import os


class QDMPlayButton(QSvgWidget):
    def __init__(self, timeline):
        super().__init__()
        self.render = self.renderer()
        self.load(asset_path('stop.svg'))
        self.timeline = timeline
        self.checked = True
        # PySide2 >= 5.15
        setKeepAspect(self.render)
    
    def isChecked(self):
        return self.checked
    
    def change(self):
        self.checked = not self.checked
        if self.checked:
            self.load(asset_path('stop.svg'))
        else:
            self.load(asset_path('play.svg'))
        setKeepAspect(self.render)
        self.timeline.value_changed()

    def mousePressEvent(self, event):
        super().mouseMoveEvent(event)
        self.change()


class QDMPrevNextButton(QSvgWidget):
    def __init__(self, timeline):
        super().__init__()
        self.render = self.renderer()
        self.load(asset_path(self.svg_up))
        self.timeline = timeline
        # PySide2 >= 5.15
        setKeepAspect(self.render)

        self.counter = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timeout)
    
    def mousePressEvent(self, event):
        super().mouseMoveEvent(event)
        self.callback()
        self.load(asset_path(self.svg_down))
        setKeepAspect(self.render)
        self.counter = 0
        self.timer.start(100)
    
    def mouseReleaseEvent(self, event):
        self.load(asset_path(self.svg_up))
        setKeepAspect(self.render)
        self.timer.stop()

    def on_timeout(self):
        self.counter += 1
        if self.counter >= 3:
            self.callback()

class QDMPrevButton(QDMPrevNextButton):
    svg_up = 'prev.svg'
    svg_down = 'prev-click.svg'

    def callback(self):
        self.timeline.prev_frame()

class QDMNextButton(QDMPrevNextButton):
    svg_up = 'next.svg'
    svg_down = 'next-click.svg'

    def callback(self):
        self.timeline.next_frame()

class QDMSlider(QSlider):
    def __init__(self, type, timeline):
        super().__init__(type)
        self.timeline = timeline
        self.valueChanged.connect(self.valueChanged_callback)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.timeline.stop_play()
        value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())
        self.setValue(value)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.timeline.stop_play()
        value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())
        self.setValue(value)

    def valueChanged_callback(self, v):
        self.timeline.label.setText(str(v))


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        validator = QIntValidator()
        validator.setBottom(0)
        self.maxframe = QLineEdit(self)
        self.maxframe.setValidator(validator)
        self.maxframe.setText('100')
        if os.environ.get('ZEN_OPEN'):
            self.maxframe.setText('1')
        self.maxframe.setFixedWidth(40)

        self.always_run = QCheckBox('Always', self)
        self.always_run.setFixedWidth(65)
        self.button_execute = QPushButton('Run', self)
        self.button_execute.setFixedWidth(40)
        self.button_kill = QPushButton('Kill', self)
        self.button_kill.setFixedWidth(40)

        self.label = QLabel('0')
        self.status = QLabel('-')

        self.slider = QDMSlider(Qt.Horizontal, self)
        self.slider.valueChanged.connect(self.value_changed)
        self.slider.sliderPressed.connect(self.stop_play)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)

        self.player = QDMPlayButton(self)
        self.prevBtn = QDMPrevButton(self)
        self.nextBtn = QDMNextButton(self)

        layout = QHBoxLayout()
        layout.addWidget(self.maxframe)
        layout.addWidget(self.always_run)
        layout.addWidget(self.button_execute)
        layout.addWidget(self.button_kill)
        layout.addWidget(self.player)
        layout.addWidget(self.prevBtn)
        layout.addWidget(self.nextBtn)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.status)
        self.setLayout(layout)
        self.setFixedHeight(38)

        self.initShortcuts()

    def initShortcuts(self):
        self.msgPlay = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.msgPlay.activated.connect(lambda: self.player.change())
        self.msgRun = QShortcut(QKeySequence(Qt.Key_A), self)
        self.msgRun.activated.connect(self.on_execute)

        self.maxframe.textChanged.connect(self.maxframe_changed)
        self.maxframe_changed()
        self.button_kill.clicked.connect(self.on_kill)
        self.button_execute.clicked.connect(self.on_execute)
        self.always_run.stateChanged.connect(self.on_always_run)

    def on_always_run(self, state):
        self.editor.always_run = state == 2
        self.editor.try_run_this_frame()

    def on_kill(self):
        self.editor.on_kill()

    def on_execute(self):
        self.editor.on_kill()
        self.editor.on_execute()
        self.slider.setValue(0)
        self.start_play()

    def setEditor(self, editor):
        self.editor = editor
        editor.edit_nframes = self.maxframe

    def maxframe_changed(self):
        self.slider.setMaximum(int('0' + self.maxframe.text()))

    def on_update(self):
        frameid = zenvis.get_curr_frameid()
        self.slider.setValue(frameid)
        self.label.setText(str(frameid))
        self.status.setText(self.get_status_string())

    def value_changed(self):
        zenvis.set_curr_frameid(self.slider.value())
        zenvis.status['playing'] = self.player.isChecked()

    def start_play(self):
        if not self.player.isChecked():
            self.player.change()
        self.value_changed()

    def stop_play(self):
        if self.player.isChecked():
            self.player.change()
        self.value_changed()

    def get_status_string(self):
        fps = zenvis.status['render_fps']
        spf = zenvis.status['solver_interval']
        return f'{fps:.1f} FPS | {spf:.02f} secs/step'

    def next_frame(self):
        self.stop_play()
        f = zenvis.get_curr_frameid()
        zenvis.set_curr_frameid(f + 1)

    def prev_frame(self):
        self.stop_play()
        f = zenvis.get_curr_frameid()
        zenvis.set_curr_frameid(f - 1)

    def jump_frame(self, frameid):
        self.stop_play()
        self.slider.setValue(frameid)
        zenvis.set_curr_frameid(frameid)
