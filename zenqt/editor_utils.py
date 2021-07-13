from .editor import *

def fillRect(painter, rect, color, line_width=None, line_color=None):
    if line_width:
        pen = QPen(QColor(line_color))
        pen.setWidth(line_width)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
    else:
        painter.setPen(Qt.NoPen)

    painter.setBrush(QColor(color))

    pathTitle = QPainterPath()
    pathTitle.addRect(rect)
    painter.drawPath(pathTitle.simplified())

MAX_STACK_LENGTH = 100

class HistoryStack:
    def __init__(self, scene):
        self.scene = scene
        self.init_state()
    
    def init_state(self):
        self.current_pointer = -1
        self.stack = []

    def undo(self):
        # can not undo at stack bottom
        if self.current_pointer == 0:
            return

        self.current_pointer -= 1
        current_scene = self.stack[self.current_pointer]
        self.scene.newGraph()
        self.scene.loadGraph(current_scene)

    def redo(self):
        # can not redo at stack top
        if self.current_pointer == len(self.stack) - 1:
            return

        self.current_pointer += 1
        current_scene = self.stack[self.current_pointer]
        self.scene.newGraph()
        self.scene.loadGraph(current_scene)
    
    def record(self):
        if self.current_pointer != len(self.stack) - 1:
            self.stack = self.stack[:self.current_pointer + 1]

        nodes = self.scene.dumpGraph()
        self.stack.append(nodes)
        self.current_pointer += 1

        # limit the stack length
        if self.current_pointer > MAX_STACK_LENGTH:
            self.stack = self.stack[1:]
            self.current_pointer = MAX_STACK_LENGTH
