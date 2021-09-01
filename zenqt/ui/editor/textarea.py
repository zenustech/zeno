from . import *


class QDMGraphicsNode_MakeMultilineString(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.width *= 1.5

        h = - TEXT_HEIGHT / 2
        offset = style['dummy_socket_offset']
        self.dummy_output_socket.setPos(self.width + offset, h)
