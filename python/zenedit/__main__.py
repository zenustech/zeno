import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .ui import QDMNodeEditorWidget, QDMGraphicsScene
from .ctx import ExecutionContext


app = QApplication(sys.argv)

win = QDMNodeEditorWidget()
scene = QDMGraphicsScene()
win.setScene(scene)

ctx = ExecutionContext()

scene.setDescriptors(ctx.get_descriptors())

node = scene.makeNode('ReadObjMesh')
node.setPos(-100, -100)
scene.addNode(node)

win.show()
sys.exit(app.exec_())
