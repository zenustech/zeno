import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .ui import QDMNodeEditorWidget
from .ctx import ExecutionContext


app = QApplication(sys.argv)

win = QDMNodeEditorWidget()
ctx = ExecutionContext()

win.setDescriptors(ctx.get_descriptors())

node = win.makeNode('ReadObjMesh')
node.setPos(-100, 100)
win.addNode(node)

win.show()
sys.exit(app.exec_())
