import sys
import random
from PySide2 import QtCore
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from . import zenvis

def lerp(_from, _to, t):
    return _from + (_to - _from) * t

class CameraKeyframeWidget(QWidget):
    def __init__(self, display):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.display = display

        self.setGeometry(300, 300, 500, 400)
        self.setWindowTitle('Camera Keyframe')

        self.keyframes = zenvis.status['camera_keyframes']

        self.list = QListWidget()

        self.enable = QCheckBox('Enable Camera Keyframe')
        self.enable.setCheckState(Qt.Checked)

        self.key = QPushButton('Key Current Frame')
        self.key.clicked.connect(self.insert_keyframe)

        self.remove = QPushButton('Remove')
        self.remove.clicked.connect(self.remove_keyframe)


        dlgLayout = QVBoxLayout()
        dlgLayout.addWidget(self.enable)
        dlgLayout.addWidget(self.list)
        dlgLayout.addWidget(self.key)
        dlgLayout.addWidget(self.remove)

        self.setLayout(dlgLayout)
    
    def checkbox_callback(self, status):
        print(satus)
        self.enable_flag = False if status == 0 else True
    
    def query_frame(self, frame):
        if self.enable.checkState() == Qt.Unchecked:
            return None
        if self.keyframes == {}:
            return None
        if frame in self.keyframes:
            return self.keyframes[frame]
        else:
            prev = [f for f in self.keyframes.keys() if f < frame]
            nxt = [f for f in self.keyframes.keys() if f > frame]
            if prev == []:
                return None
            elif nxt == []:
                p = prev[-1]
                return self.keyframes[p]
            else:
                p = prev[-1]
                pv = self.keyframes[p]

                n = nxt[0]
                nv = self.keyframes[n]

                t = (frame - p) / (n - p)
                r = []
                for i in range(len(pv)):
                    r.append(lerp(pv[i], nv[i], t))
                
                return tuple(r)

    def insert_keyframe(self):
        frameid = zenvis.get_curr_frameid()
        self.keyframes[frameid] = zenvis.status['perspective']
        self.update_list()

    def remove_keyframe(self):
        sel = self.list.currentRow()
        if sel < 0:
            return
        l = sorted(list(self.keyframes.keys()))
        k = l[sel]
        self.keyframes.pop(k)
        self.update_list()
    
    def update_list(self):
        self.list.clear()
        l = sorted(list(self.keyframes.keys()))
        self.list.addItems(map(str, l))
    
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        self.update_list()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Win()
    form.show()
    sys.exit(app.exec_())

