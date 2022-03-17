import sys
import math
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from typing import Union, Tuple


point_size = 7
dist_len = 5


ZRed = QColor(204, 38, 38)
ZGreen = QColor(38, 204, 38)
ZBlue = QColor(38, 38, 204)
ZGray = QColor(127, 127, 127)
ZYellow = QColor(255, 255, 204)

def clamp(v, min_v, max_v):
    if v < min_v:
        return min_v
    elif v > max_v:
        return max_v
    else:
        return v

class ValuePoint:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __add__(self, rhs):
        v = ValuePoint(self.x + rhs.x, self.y + rhs.y)
        return v

    def __sub__(self, rhs):
        v = ValuePoint(self.x - rhs.x, self.y - rhs.y)
        return v

    def __mul__(self, rhs):
        v = ValuePoint(self.x * rhs, self.y * rhs)
        return v

    def len(self):
        v = math.sqrt(self.x * self.x + self.y * self.y)
        return v

    def norm(self):
        l = self.len()
        x = self.x / l
        y = self.y / l
        return ValuePoint(x, y)

    def lerp(self, to, t):
        v = self + (to - self) * t
        return v

    def flip(self):
        v = ValuePoint(-self.x, -self.y)
        return v

class Bezier:
    def __init__(self, p1: ValuePoint, p2: ValuePoint, h1: ValuePoint, h2: ValuePoint):
        self.p1 = p1
        self.p2 = p2
        self.h1 = h1
        self.h2 = h2
    
    def calc(self, t: float) -> ValuePoint:
        assert(0.0 <= t and t <= 1.0)
        a = self.p1.lerp(self.h1, t)
        b = self.h1.lerp(self.h2, t)
        c = self.h2.lerp(self.p2, t)
        d = a.lerp(b, t)
        e = b.lerp(c, t)
        f = d.lerp(e, t)
        return f
    
    def query(self, x: float) -> float:
        epsilon = 0.00001
        lower = 0
        upper = 1
        left_calc_count = 100
        t = (lower + upper) / 2
        np = self.calc(t)

        while abs(np.x - x) > epsilon and left_calc_count > 0:
            assert lower < t and t < upper
            if x < np.x:
                upper = t
            else:
                lower = t
            t = (lower + upper) / 2
            np = self.calc(t)
            left_calc_count -= 1
        # assert left_calc_count > 0
        return np.y

class ControlPoint:
    def __init__(self, x, y) -> None:
        self.pos = ValuePoint(x, y)
        self.left_handler = ValuePoint(-5, 0)
        self.right_handler = ValuePoint(5, 0)
        # straight, constant, align, free
        self.cp_type = 'align'

class MainCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.middle_pressed = False
        self.left_pressed = False
        self.global_pos = QPoint(0, 0)

        self.scale_x = 5
        self.scale_y = 100
        self.offset_x = 0
        self.offset_y = 0

        self.scale_level = 0
        self.scale_base = 1.5

        self.selected = None

        self.cur_frame = 0

    def paintEvent(self, e):
        super().paintEvent(e)
        w = self.width()
        h = self.height()
        qp = QPainter()
        qp.begin(self)
        
        qp.drawLine(
            0,
            self.offset_y + h // 2,
            w,
            self.offset_y + h // 2
        )

        self.draw_select_time(qp)
        self.draw_curves(qp)
        self.draw_keypoints(qp)
        qp.end()
    
    def draw_select_time(self, qp):
        p = self.raw_ro_canvas(ValuePoint(self.cur_frame, 0))
        qp.drawLine(
            p.x(),
            0,
            p.x(),
            self.height()
        )

    def mouseMoveEvent(self, e) -> None:
        super().mouseMoveEvent(e)
        p = e.globalPos()
        if self.middle_pressed:
            self.move_view(p)
        elif self.left_pressed:
            if self.get_selected() != None:
                self.move_point(e.pos())
            else:
                pos = e.pos()
                self.sync_cur_frame(pos)

        self.global_pos = p
        self.curve_wnd.update()

    def mousePressEvent(self, e) -> None:
        super().mousePressEvent(e)
        self.setMouseTracking(True)
        self.global_pos = e.globalPos()
        if e.button() ==  Qt.MiddleButton:
            self.middle_pressed = True
        elif e.button() ==  Qt.LeftButton:
            self.left_pressed = True
            pos = e.pos()
            sel = self.select_keypoint(pos)
            self.set_selected(sel)
            if sel == None:
                self.sync_cur_frame(pos)
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        super().mouseReleaseEvent(e)
        self.setMouseTracking(False)
        if e.button() ==  Qt.MiddleButton:
            self.middle_pressed = False
        elif e.button() ==  Qt.LeftButton:
            self.left_pressed = False

    def wheelEvent(self, e) -> None:
        super().wheelEvent(e)
        if e.angleDelta().y() > 0:
            self.scale_level += 1
        else:
            self.scale_level -= 1
        
        self.update()
    
    def raw_ro_canvas(self, p: ValuePoint) -> QPoint:
        h = self.height()
        x = p.x * self.scale_x + self.offset_x
        y = h // 2 - p.y * self.scale_y * (self.scale_base ** self.scale_level) + self.offset_y
        np = QPoint(x, int(y))
        return np

    def canvas_to_raw(self, p: QPoint) -> ValuePoint:
        h = self.height()
        x = (p.x() - self.offset_x) / self.scale_x
        y = (p.y() - h // 2 - self.offset_y) / (self.scale_y * (self.scale_base ** self.scale_level)) * -1 
        return ValuePoint(x, y)
    
    def draw_curves(self, qp):
        qp.setPen(ZRed)
        for i in range(0, len(self.get_ps())-1):
            prev = self.raw_ro_canvas(self.get_ps()[i].pos)
            nxt = self.raw_ro_canvas(self.get_ps()[i+1].pos)
            if self.get_ps()[i].cp_type == 'constant':
                qp.drawLine(prev.x(), prev.y(), nxt.x(), prev.y())
            elif self.get_ps()[i].cp_type == 'straight':
                qp.drawLine(prev.x(), prev.y(), nxt.x(), nxt.y())
            else:
                path = QPainterPath(prev)
                lp = self.raw_ro_canvas(self.get_ps()[i].pos + self.get_ps()[i].right_handler)
                rp = self.raw_ro_canvas(self.get_ps()[i+1].pos + self.get_ps()[i+1].left_handler)
                lp.setX(clamp(lp.x(), prev.x(), nxt.x()))
                rp.setX(clamp(rp.x(), prev.x(), nxt.x()))
                path.cubicTo(lp, rp, nxt)
                qp.drawPath(path)

    def draw_keypoints(self, qp):
        psize = point_size
        hpsize = psize // 2
        for i in range(len(self.get_ps())):
            p = self.get_ps()[i]
            qp.setPen(ZGray)
            np = self.raw_ro_canvas(p.pos)
            if i < len(self.get_ps()) - 1 and p.cp_type in ['align', 'free']:
                rp = self.raw_ro_canvas(p.pos + p.right_handler)
                qp.fillRect(rp.x() - hpsize, rp.y() - hpsize, psize, psize, ZGray)
                qp.drawLine(np.x(), np.y(), rp.x(), rp.y())

            if i > 0 and self.get_ps()[i - 1].cp_type in ['align', 'free']:
                p = self.get_ps()[i]
                lp = self.raw_ro_canvas(p.pos + p.left_handler)
                qp.fillRect(lp.x() - hpsize, lp.y() - hpsize, psize, psize, ZGray)
                qp.drawLine(np.x(), np.y(), lp.x(), lp.y())

            qp.fillRect(np.x() - hpsize, np.y() - hpsize, psize, psize, ZYellow if i == self.get_selected() else ZGray)
                

    def select_keypoint(self, pos: QPoint) -> Union[int, Tuple[int, int], None]:
        for i in range(len(self.get_ps())):
            p = self.raw_ro_canvas(self.get_ps()[i].pos)
            dist = QPoint(p.x() - pos.x(), p.y() - pos.y())
            if dist.manhattanLength() < dist_len:
                return i
            if i < len(self.get_ps()) - 1 and self.get_ps()[i].cp_type in ['align', 'free']:
                p = self.get_ps()[i]
                rp = self.raw_ro_canvas(p.pos + p.right_handler)
                dist = QPoint(rp.x() - pos.x(), rp.y() - pos.y())
                if dist.manhattanLength() < dist_len:
                    return i, 1

            if i > 0 and self.get_ps()[i - 1].cp_type in ['align', 'free']:
                p = self.get_ps()[i]
                lp = self.raw_ro_canvas(p.pos + p.left_handler)
                dist = QPoint(lp.x() - pos.x(), lp.y() - pos.y())
                if dist.manhattanLength() < dist_len:
                    return i, 0
        return None
    
    def move_view(self, gp):
        delta_x = gp.x() - self.global_pos.x()
        delta_y = gp.y() - self.global_pos.y()
        self.offset_x += delta_x
        self.offset_y += delta_y
    
    def move_point(self, p: QPoint):
        raw_p = self.canvas_to_raw(p)
        s = self.get_selected()
        if type(s) == int:
            self.get_ps()[s].pos.y = raw_p.y
            target_x = round(raw_p.x)
            if s > 0:
                target_x = max(self.get_ps()[s - 1].pos.x + 1, target_x)
            if s < len(self.get_ps()) - 2:
                target_x = min(self.get_ps()[s + 1].pos.x - 1, target_x)
            if s == 0:
                target_x = 0
            self.get_ps()[s].pos.x = target_x
        else:
            sel_p = self.get_ps()[s[0]]
            if s[1] == 0:
                sel_p.left_handler = raw_p - sel_p.pos
                if sel_p.cp_type == 'align':
                    sel_p.right_handler = sel_p.left_handler.flip().norm() * sel_p.right_handler.len()
            else:
                sel_p.right_handler = raw_p - sel_p.pos
                if sel_p.cp_type == 'align':
                    sel_p.left_handler = sel_p.right_handler.flip().norm() * sel_p.left_handler.len()

    def constant(self):
        s = self.get_selected()
        if type(s) == int:
            self.get_ps()[s].cp_type = 'constant'
            self.curve_wnd.update()

    def straight(self):
        s = self.get_selected()
        if type(s) == int:
            self.get_ps()[s].cp_type = 'straight'
            self.curve_wnd.update()

    def free(self):
        s = self.get_selected()
        if type(s) == int:
            self.get_ps()[s].cp_type = 'free'
            self.curve_wnd.update()

    def align(self):
        s = self.get_selected()
        if type(s) == int:
            self.get_ps()[s].cp_type = 'align'
            self.get_ps()[s].left_handler = ValuePoint(-5, 0)
            self.get_ps()[s].right_handler = ValuePoint(5, 0)
            self.curve_wnd.update()

    def sync_cur_frame(self, pos: QPoint):
        self.cur_frame = int(self.canvas_to_raw(pos).x)
        self.curve_wnd.widget_state.set_cur_frame(self.cur_frame)

    def quary_value(self, x: float) -> float:
        s = self.curve_wnd.widget_state
        return s.quary_value(x, s.sel_channel)

    def set_keyframe_x_value(self, target_x: int):
        s = self.get_selected()
        if type(s) == int:
            if s > 0:
                target_x = max(self.get_ps()[s - 1].pos.x + 1, target_x)
            if s < len(self.get_ps()) - 2:
                target_x = min(self.get_ps()[s + 1].pos.x - 1, target_x)
            if s == 0:
                target_x = 0
            self.get_ps()[s].pos.x = target_x
            self.update()

    def set_keyframe_y_value(self, target_y: float):
        s = self.get_selected()
        if type(s) == int:
            self.get_ps()[s].pos.y = target_y
            self.update()

    def get_ps(self):
        s = self.curve_wnd.widget_state
        ps = s.data[s.sel_channel]
        return ps

    def get_selected(self):
        s = self.curve_wnd.widget_state
        return s.sel_point_index

    def set_selected(self, selected):
        self.selected = selected
        s = self.curve_wnd.widget_state
        s.sel_point_index = selected
        self.curve_wnd.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # s = CurveGraphicsScene()
    # for x, y in ps:
    #     s.addItem(ControlPoint(x * scale, y * scale))

    # w = CurveGraphicsView()
    # w.setScene(s)
    w = MainCanvas()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    sys.exit(app.exec_())
