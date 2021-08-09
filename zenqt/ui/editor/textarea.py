from . import *


class QDMGraphicsNode_MakeMultilineString(QDMGraphicsNode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.width *= 1.5

