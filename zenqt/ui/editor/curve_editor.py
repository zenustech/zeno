from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
import sys, random, math
from copy import deepcopy

radius = 5
bound = 20
distance_threshold = 0.05
segment = 20
epsilon = 0.00001

def lerp(start, end, t):
    return start + t * (end - start)

def plerp(p1, p2, t):
    return (
        lerp(p1[0], p2[0], t),
        lerp(p1[1], p2[1], t),
    )

def padd(p1, p2):
    return (
        p1[0] + p2[0],
        p1[1] + p2[1],
    )

def psub(p1, p2):
    return (
        p1[0] - p2[0],
        p1[1] - p2[1],
    )

def pdist(p1, p2):
    x = abs(p1[0] - p2[0])
    y = abs(p1[1] - p2[1])
    return math.sqrt(x * x + y * y)

def plen(p):
    x = abs(p[0])
    y = abs(p[1])
    return math.sqrt(x * x + y * y)

def pnorm(p):
    length = plen(p)
    return (
        p[0] / length,
        p[1] / length,
    )

def pmul(p, f):
    return (
        p[0] * f,
        p[1] * f,
    )

def bezier(p1, p2, h1, h2, t):
    a = plerp(p1, h1, t)
    b = plerp(h1, h2, t)
    c = plerp(h2, p2, t)
    d = plerp(a, b, t)
    e = plerp(b, c, t)
    f = plerp(d, e, t)
    return f

def eval_value(points, handlers, x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    i = len(list(filter(lambda p: p[0] < x, points))) - 1
    p1 = points[i]
    p2 = points[i + 1]
    h1 = handlers[i][1]
    h2 = handlers[i+1][0]
    return eval_bezier_value(p1, p2, h1, h2, x)

def eval_bezier_value(p1, p2, h1, h2, x):
    lower = 0
    upper = 1
    left_calc_count = 100
    t = (lower + upper) / 2
    np = bezier(p1, p2, h1, h2, t)

    while abs(np[0] - x) >  epsilon and left_calc_count > 0:
        assert lower < t and t < upper
        if x < np[0]:
            upper = t
        else:
            lower = t
        t = (lower + upper) / 2
        np = bezier(p1, p2, h1, h2, t)
        left_calc_count -= 1
    assert left_calc_count > 0
    return np[1]

def clamp(_input, _min, _max):
    if _input < _min:
        return _min
    elif _input > _max:
        return _max
    else:
        return _input

class CurveEditor(QDialog):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.points = node.points
        self.handlers = node.handlers
        self.params = node.params

        self.initUI()
        
    def initUI(self):
        self.idx = None
        self.pressed = False
        self.alt_pressed = False
        self.w = 300
        self.h = 200
        self.length = min(self.w, self.h)
        self.setGeometry(300, 300, self.w, self.h)
        self.setWindowTitle('CurveEditor')
        self.show()

    def paintEvent(self, e):
        points = self.points

        self.length = min(self.width(), self.height()) - bound * 2
        qp = QPainter()
        qp.begin(self)
        qp.setPen(Qt.gray)
        self.drawLines(qp, points)
        qp.setPen(Qt.magenta)
        self.drawHandlers(qp, self.idx)
        qp.setPen(Qt.blue)
        self.drawPoints(qp, points, self.idx)
        if type(self.idx) == int:
            p = self.points[self.idx]
            info = 'x={}, y={}'.format(p[0], p[1])
            qp.drawText(self.length + 2 * bound, bound, info)
            info2 = 'input={}, output={}'.format(
                lerp(self.params['input_min'].getValue(), self.params['input_max'].getValue(), p[0]),
                lerp(self.params['output_min'].getValue(), self.params['output_max'].getValue(), p[1]),
            )
            qp.drawText(self.length + 2 * bound, bound * 2, info2)
        qp.end()

    def drawPoints(self, qp, points, sel=None):
        if type(sel) == tuple:
            sel = sel[0]
        for i in range(len(points)):
            p = points[i]
            x = int(self.length * p[0]) - radius + bound
            y = int(self.length * (1 - p[1])) - radius + bound
            if i == sel:
                qp.fillRect(x, y, 10, 10, qp.pen().color())
            else:
                qp.drawRect(x, y, 10, 10)

    
    def drawPolyline(self, qp, points):
        for i in range(len(points) - 1):
            p1 = points[i]
            x1 = int(self.length * p1[0]) + bound
            y1 = int(self.length * (1 - p1[1])) + bound
            p2 = points[i + 1]
            x2 = int(self.length * p2[0]) + bound
            y2 = int(self.length * (1 - p2[1])) + bound
            qp.drawLine(x1, y1, x2, y2)

    def drawLines(self, qp, points):
        handlers = self.handersToPoints(self.correct_handlers())

        ps = []
        for i in range(len(points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            h1 = handlers[i][1]
            h2 = handlers[i+1][0]
            for t in range(segment + 1):
                t = t / segment
                ps.append(bezier(p1, p2, h1, h2, t))
        self.drawPolyline(qp, ps)
    
    def drawHandlers(self, qp, sel):
        h = None
        if type(sel) == tuple:
            h = sel[1]
            sel = sel[0]
        if sel == None:
            return
        handlers = self.handersToPoints(self.handlers)
        hs = handlers[sel]
        if sel != 0 and sel != len(self.points) - 1:
            self.drawPoints(qp, hs, h)
            self.drawPolyline(qp, [hs[0], self.points[sel], hs[1]])
        elif sel == 0:
            self.drawPoints(qp, hs[1:], 0 if h != None else None)
            self.drawPolyline(qp, [self.points[sel], hs[1]])
        elif sel == len(self.points) - 1:
            self.drawPoints(qp, hs[:1], 0 if h != None else None)
            self.drawPolyline(qp, [hs[0], self.points[sel]])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.alt_pressed = True
        if event.key() == Qt.Key_Delete and self.idx and len(self.points) >= 3:
            if type(self.idx) == int:
                self.points.pop(self.idx)
                self.idx = None
            else:
                i = self.idx
                self.handlers[i[0]][i[1]] = (0, 0)
                self.idx = i[0]
            self.update()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.alt_pressed = False

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        if event.button() == Qt.LeftButton:
            self.pressed = True
            if x < self.length + 2 * bound and y < self.length + 2 * bound:
                x = (x - bound) / self.length
                y = 1 - (y - bound) / self.length
                sel_handler = self.near_handler(x, y)
                if sel_handler != None:
                    self.idx = sel_handler
                else:
                    sel_point = self.near_point(x, y)
                    if sel_point == None:
                        sel_point = len(list(filter(lambda p: p[0] < x, self.points)))
                        self.points.insert(sel_point, (x, y))
                        v = psub(self.points[sel_point + 1], self.points[sel_point - 1])
                        self.handlers.insert(sel_point, [
                                pmul(v, - 1 / 6),
                                pmul(v, 1 / 6),
                            ]
                        )
                    self.idx = sel_point
            else:
                self.idx = None
            self.update()
        elif event.button() == Qt.MiddleButton:
            if bound < x and x < self.length + bound and bound < y and y < self.length + bound:
                x = (x - bound) / self.length
                x = clamp(x, 0, 1)
                handlers = self.handersToPoints(self.correct_handlers())
                print(eval_value(self.points, handlers, x))

    def mouseReleaseEvent(self, event):
        self.pressed = False

    def mouseMoveEvent(self, event):
        if self.idx != None and self.pressed:
            x = (event.x() - bound) / self.length
            y = (event.y() - bound) / self.length
            if type(self.idx) == int:
                if self.idx == 0:
                    x = 0
                elif self.idx == len(self.points) - 1:
                    x = 1
                else:
                    x = clamp(x, self.points[self.idx - 1][0], self.points[self.idx + 1][0])
                y = 1 - clamp(y, 0, 1)
                self.points[self.idx] = (x, y)
            elif type(self.idx) == tuple:
                i = self.idx
                x, y = psub((x, 1 - y), self.points[self.idx[0]])
                if i[1] == 0:
                    x = x if x < 0 else 0
                else:
                    x = x if x > 0 else 0
                self.handlers[i[0]][i[1]] = (x, y)
                if not self.alt_pressed:
                    v = pnorm((x, y))
                    self.handlers[i[0]][1-i[1]] = pmul(
                        v,
                        plen(self.handlers[i[0]][1-i[1]]) * -1,
                    )

            self.update()

    def near_point(self, x, y):
        idx = None
        min_dist = 1
        for i in range(len(self.points)):
            p = self.points[i]
            dist = pdist((x, y), p)
            if dist < min_dist:
                idx = i
                min_dist = dist
        return idx if min_dist < distance_threshold else None

    def near_handler(self, x, y):
        idx = None
        min_dist = 1
        for i in range(len(self.points)):
            for j in range(2):
                if i == 0 and j == 0:
                    continue
                if i == len(self.points) - 1 and j == 1:
                    continue
                
                h = padd(self.points[i], self.handlers[i][j])
                dist = pdist((x, y), h)
                if dist < min_dist:
                    idx = (i, j)
                    min_dist = dist
        return idx if min_dist < distance_threshold else None

    # correctness modify: make bezier to be function
    def correct_handlers(self):
        handlers = deepcopy(self.handlers)
        for i in range(len(self.points)):
            cur_p = self.points[i]
            if i != 0:
                hp = handlers[i][0]
                prev_p = self.points[i-1]
                if abs(hp[0]) > abs(cur_p[0] - prev_p[0]):
                    s = abs(cur_p[0] - prev_p[0]) / abs(hp[0])
                    handlers[i][0] = pmul(hp, s)

            if i != len(self.points) - 1:
                hn = handlers[i][1]
                next_p = self.points[i+1]
                if abs(hn[0]) > abs(cur_p[0] - next_p[0]):
                    s = abs(cur_p[0] - next_p[0]) / abs(hn[0])
                    handlers[i][1] = pmul(hn, s)

        return handlers

    def handersToPoints(self, handlers):
        ps = []
        for i in range(len(self.points)):
            ps.append([
                padd(self.points[i], handlers[i][0]),
                padd(self.points[i], handlers[i][1]),
            ])
        return ps
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
