from .editor import *

def fillRect(painter, rect, color, line_width=None, line_color=None):
    painter.setPen(Qt.NoPen)
    if line_width:
        painter.fillRect(rect, QColor(line_color))

        w = line_width
        r = rect
        content_rect = QRect(r.x() + w, r.y() + w, r.width() - w * 2, r.height() - w * 2)
        painter.fillRect(content_rect, QColor(color))
    else:
        painter.fillRect(rect, QColor(color))


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
