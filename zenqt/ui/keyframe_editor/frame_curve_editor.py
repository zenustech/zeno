from re import L
import sys
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

from .curve_canvas import MainCanvas, ControlPoint, Bezier

half_len = 2

def lerp(start, end, t):
    return start + t * (end - start)

class ToolBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.btn_constant = QPushButton('Constant')
        self.btn_straight = QPushButton('Straight')
        self.btn_align = QPushButton('Align')
        self.btn_free = QPushButton('Free')
        self.btn_add = QPushButton('Add')
        self.btn_delete = QPushButton('Delete')
        self.frame_value = QLabel('  0')

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.frame_value)
        self.layout.addWidget(self.btn_align)
        self.layout.addWidget(self.btn_free)
        self.layout.addWidget(self.btn_constant)
        self.layout.addWidget(self.btn_straight)
        self.layout.addWidget(self.btn_add)
        self.layout.addWidget(self.btn_delete)
        self.setLayout(self.layout)

        self.setFixedHeight(50)

    def paintEvent(self, e):
        super().paintEvent(e)
        f = self.curve_wnd.widget_state.cur_frame
        self.frame_value.setText('{:3d}'.format(f))


class ChannelsWidget(QGroupBox):
    def __init__(self, state) -> None:
        super().__init__()
        self.widget_state = state
        self.lst = list(state.data.keys())


        self.radio_btns = {}
        self.value_labels = {}
        for name in self.lst:
            self.radio_btns[name] = QRadioButton(name)
            self.radio_btns[name].clicked.connect(self.set_sel_channel)
            self.value_labels[name] = QLabel('{:.3f}'.format(0))

        self.radio_btns[self.lst[0]].setChecked(True)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        for i, k in enumerate(self.radio_btns):
            self.layout.addWidget(self.radio_btns[k], i, 0)
            self.layout.addWidget(self.value_labels[k], i, 1)

        self.setTitle('Channels')

    def set_cur_value(self):
        s = self.curve_wnd.widget_state
        f = self.curve_wnd.widget_state.cur_frame
        for name in self.lst:
            self.value_labels[name].setText('{:.3f}'.format(s.query_value(f, name)))

    def paintEvent(self, e):
        super().paintEvent(e)
        self.set_cur_value()
    
    def set_sel_channel(self):
        for name in self.lst:
            if self.radio_btns[name].isChecked():
                self.widget_state.set_sel_channel(name)

class CurrentPanelWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__()
        self.label_x = QLabel('x: ')
        self.label_y = QLabel('y: ')
        self.edit_x = QLineEdit()
        self.edit_y = QLineEdit()

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.label_x, 0, 0)
        self.layout.addWidget(self.label_y, 1, 0)
        self.layout.addWidget(self.edit_x, 0, 1)
        self.layout.addWidget(self.edit_y, 1, 1)

        self.setTitle('Point')

    def paintEvent(self, e):
        super().paintEvent(e)
        s = self.curve_wnd.widget_state
        if s.editing == False:
            if type(s.sel_point_index) == int:
                p = s.data[s.sel_channel][s.sel_point_index]
                self.edit_x.setText(str(p.pos.x))
                self.edit_y.setText('{:.3f}'.format(p.pos.y))
            else:
                self.edit_x.setText('')
                self.edit_y.setText('')


class LeftColumn(QWidget):
    def __init__(self, state) -> None:
        super().__init__()
        self.channels_widget = ChannelsWidget(state)
        self.current_panel = CurrentPanelWidget()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.channels_widget)
        self.layout.addWidget(self.current_panel)
        self.setFixedWidth(200)

class RightColumn(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.tool_bar = ToolBar()
        self.main_canvas = MainCanvas()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.tool_bar)
        self.layout.addWidget(self.main_canvas)
        self.bind()
    
    def bind(self):
        self.tool_bar.btn_constant.clicked.connect(lambda : self.main_canvas.constant())
        self.tool_bar.btn_straight.clicked.connect(lambda : self.main_canvas.straight())
        self.tool_bar.btn_align.clicked.connect(lambda : self.main_canvas.align())
        self.tool_bar.btn_free.clicked.connect(lambda : self.main_canvas.free())

class CurveWindowState:
    def __init__(self, widget, data) -> None:
        self.widget = widget 
        self.sel_channel = next(iter(data))
        self.data = data
        self.cur_frame = 0
        self.sel_point_index = None
        self.editing = False

    def set_sel_channel(self, sel_channel: str):
        if self.sel_channel == sel_channel:
            return
        self.sel_channel = sel_channel
        self.sel_point_index = None
        self.widget.update()

    def set_cur_frame(self, cur_frame: int):
        self.cur_frame = cur_frame
        self.widget.update()

    def query_value(self, x: float, sel_channel: str) -> float:
        ps = self.data[sel_channel]
        if len(ps) == 0:
            return 0
        if x < ps[0].pos.x or x > ps[-1].pos.x:
            return 0
        i = len(list(filter(lambda p: p.pos.x <= x, ps))) - 1
        p1 = ps[i].pos
        if p1.x == x or ps[i].cp_type == 'constant':
            return p1.y
        elif ps[i].cp_type == 'straight':
            p2 = ps[i + 1].pos
            t = (x - p1.x) / (p2.x - p1.x)
            return lerp(p1.y, p2.y, t)
        else:
            p2 = ps[i + 1].pos
            h1 = ps[i].pos + ps[i].right_handler
            h2 = ps[i+1].pos + ps[i+1].left_handler
            b = Bezier(p1, p2, h1, h2)
            return b.query(x)

    def add_point(self):
        if type(self.sel_point_index) == int:
            ps = self.data[self.sel_channel]
            c = ps[self.sel_point_index].pos
            if self.sel_point_index == len(ps) - 1:
                np = ControlPoint(int(c.x + 20), c.y)
                ps.append(np)
                self.sel_point_index += 1
            else:
                n = ps[self.sel_point_index + 1].pos
                if n.x - c.x >= 2:
                    np = ControlPoint(int((c.x + n.x) / 2), (c.y + n.y) / 2)
                    ps.insert(self.sel_point_index + 1, np)
                    self.sel_point_index += 1
            self.widget.update()

    def delete_point(self):
        if type(self.sel_point_index) == int:
            ps = self.data[self.sel_channel]
            if len(ps) > 1 and self.sel_point_index > 0:
                ps.pop(self.sel_point_index)
                self.sel_point_index -= 1
                self.widget.update()

class CurveWindow(QWidget):
    def __init__(self, data) -> None:
        super().__init__()
        self.widget_state = CurveWindowState(self, data)

        self.left_column = LeftColumn(self.widget_state)
        self.right_column = RightColumn()

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.left_column)
        self.layout.addWidget(self.right_column)
        self.bind()

    def bind(self):
        self.left_column.channels_widget.curve_wnd = self
        self.left_column.current_panel.curve_wnd = self
        self.right_column.main_canvas.curve_wnd = self
        self.right_column.tool_bar.curve_wnd = self

        self.right_column.tool_bar.btn_add.clicked.connect(self.widget_state.add_point)
        self.right_column.tool_bar.btn_delete.clicked.connect(self.widget_state.delete_point)

        def edit_x_callback():
            t = self.left_column.current_panel.edit_x.text()
            self.right_column.main_canvas.set_keyframe_x_value(int(t))
            self.widget_state.editing = False
        self.left_column.current_panel.edit_x.editingFinished.connect(edit_x_callback)

        def edit_y_callback():
            t = self.left_column.current_panel.edit_y.text()
            self.right_column.main_canvas.set_keyframe_y_value(float(t))
            self.widget_state.editing = False
        self.left_column.current_panel.edit_y.editingFinished.connect(edit_y_callback)

        def editing_callback():
            self.widget_state.editing = True
        self.left_column.current_panel.edit_x.textEdited.connect(editing_callback)
        self.left_column.current_panel.edit_y.textEdited.connect(editing_callback)
        self.left_column.current_panel.edit_x.selectionChanged.connect(editing_callback)
        self.left_column.current_panel.edit_y.selectionChanged.connect(editing_callback)

ps_x = [
    ControlPoint(0, 0),
    ControlPoint(11, 0.6416),
    ControlPoint(33, 0.1583333),
    ControlPoint(56, 0.133333),
    ControlPoint(72, 0.0849),
    ControlPoint(100, 1),
]
ps_y = [
    ControlPoint(0, 0),
    ControlPoint(33, 0.583333),
    ControlPoint(56, 0.93333),
    ControlPoint(80, 0.51),
]
ps_z = [
    ControlPoint(0, 0),
]
ps_w = [
    ControlPoint(0, 0),
]
ps_h = [
    ControlPoint(0, 0),
]

data = {
    'x': ps_x,
    'y': ps_y,
    'z': ps_z,
    'w': ps_w,
    'h': ps_h,
}

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = CurveWindow(data)
    w.resize(800, 600)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())
