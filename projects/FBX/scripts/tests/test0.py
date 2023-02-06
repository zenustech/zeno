import PySide2
from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import QObject, Signal, Slot    
import maya.OpenMayaUI as mui
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import shiboken2 
import socketserver


class MayaApi(QtCore.QObject):
    def setFrame(self, frame):
        print("frame is", frame)
        for i in range(10):
            mt = om.MTime(i)
            print("set current time", i)
            oma.MAnimControl.setCurrentTime(mt)

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global maya_basicTest_window

        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        d = self.data.decode("utf-8")
        ds = d.split(' ')
        if ds[0] == "FRAME":
            maya_basicTest_window.worker.TcpSignal.emit(d)
        
        self.request.sendall(b"OK")

class Worker(QtCore.QObject):

    TcpSignal = Signal(str)

    def __init__(self):
        super(Worker, self).__init__()

    def run(self):
        print("worker run")
        with socketserver.TCPServer(("localhost", 9999), MyTCPHandler) as server:
            server.serve_forever()


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__()
        self.setMinimumSize(520, 520)
        self.api = MayaApi()

        self.runButton = QtWidgets.QPushButton()
        self.runButton.setText("Run")
        self.runButton.clicked.connect(self.runThread)

        self.testButton = QtWidgets.QPushButton()
        self.testButton.setText("Test")
        self.testButton.clicked.connect(self.onTestBtnClicked)


        self.mainLayout = QtWidgets.QVBoxLayout()
        self.centralw = QtWidgets.QWidget()
        self.centralw.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralw)

        self.mainLayout.addWidget(self.runButton)
        self.mainLayout.addWidget(self.testButton)

    def onTestBtnClicked(self):
        self.api.setFrame()

    @Slot(str)
    def onTcpSignal(self, d):
        print("onTcpSignal", d)
        ds = d.split(' ')
        if ds[0] == "FRAME":
            frame = int(ds[1])
            self.api.setFrame(frame)

    def runThread(self):
        print("run Thread")
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        self.worker.TcpSignal.connect(self.onTcpSignal)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        # Step 6: Start the thread
        self.thread.start()
        print("run Thread end")

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken2.wrapInstance(int(ptr), QtCore.QObject)

def mayaMain():
    global maya_basicTest_window
    try:
        maya_basicTest_window.close()
    except:
        pass
    print("-"*10, "Window show")

    maya_basicTest_window = Ui_MainWindow(getMayaWindow())
    maya_basicTest_window.show()

mayaMain()