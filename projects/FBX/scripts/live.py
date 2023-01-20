import sys
import time
import socket
import ctypes
import PySide2
import maya.OpenMayaUI as mui
import shiboken2 
import threading
import socketserver
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import maya.cmds as cmds
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Signal, Slot

sys.path.append(r"C:\Users\AMD\AppData\Roaming\Python\Python310\site-packages")
sys.path.append(r"C:\Users\AMD\scoop\persist\python\Lib\site-packages")
import requests


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global maya_basicTest_window

        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        d = self.data.decode("utf-8")
        maya_basicTest_window.helper.my_worker.tcpSignal.emit(d)
    
        self.request.sendall(b"OK")

class ServerWorker(QtCore.QObject):

    tcpSignal = Signal(str)
    finished = Signal()

    def __init__(self, addr, port):
        super(ServerWorker, self).__init__()
        self.addr = addr
        self.port = port
        self.server = None

    def run(self):
        print("-"*10, "TCP Server Running on", self.addr, self.port)

        self.server = socketserver.TCPServer((self.addr, int(self.port)), MyTCPHandler)
        self.server.serve_forever()

        # with socketserver.TCPServer((self.addr, int(self.port)), MyTCPHandler) as server:
        #     # Activate the server; this will keep running until you
        #     # interrupt the program with Ctrl-C
        #     server.serve_forever()

    def stop(self):
        print("-"*10, "Stop TCP Server")
        self.server.shutdown()
        self.server.server_close()
        self.finished.emit()
        print("-"*10, "Stop Finished")

class MayaApi:
    def __init__(self) -> None:
        self.vertex_count_list = []
        self.vertex_list_list = []
        self.points_list = []

    def getCurrentFrame(self):
        ct = oma.MAnimControl.currentTime()
        print("get current time", ct.value())
        return ct.value()

    def setCurrentFrame(self, frame):
        mt = om.MTime(frame)
        print("set current time", frame)
        oma.MAnimControl.setCurrentTime(mt)

    def perspCameraData(self):
        translateX = cmds.getAttr("persp.translateX")
        translateY = cmds.getAttr("persp.translateY")
        translateZ = cmds.getAttr("persp.translateZ")
        rotateX = cmds.getAttr("persp.rotateX")
        rotateY = cmds.getAttr("persp.rotateY")
        rotateZ = cmds.getAttr("persp.rotateZ")
        scaleX = cmds.getAttr("persp.scaleX")
        scaleY = cmds.getAttr("persp.scaleY")
        scaleZ = cmds.getAttr("persp.scaleZ")
        #print("translate", translateX, translateY, translateZ)
        #print("rotate", rotateX, rotateY, rotateZ)
        #print("scale", scaleX, scaleY, scaleZ)
        self.cam_trans.clear()
        self.cam_trans.append(translateX)
        self.cam_trans.append(translateY)
        self.cam_trans.append(translateZ)
        self.cam_trans.append(rotateX)
        self.cam_trans.append(rotateY)
        self.cam_trans.append(rotateZ)
        self.cam_trans.append(scaleX)
        self.cam_trans.append(scaleY)
        self.cam_trans.append(scaleZ)

    def selectionMeshData(self):
        selection = om.MSelectionList()
        om.MGlobal.getActiveSelectionList( selection )
        iterSel = om.MItSelectionList(selection, om.MFn.kMesh)
        #print("Sel ", selection)
        self.vertex_count_list = []
        self.vertex_list_list = []
        self.points_list = []

        while not iterSel.isDone():
            # get dagPath
            dagPath = om.MDagPath()
            iterSel.getDagPath( dagPath )
            print("Dag Path", dagPath)
            # create empty point array
            inMeshMPointArray = om.MPointArray()
            # create function set and get points in world space
            currentInMeshMFnMesh = om.MFnMesh(dagPath)
            currentInMeshMFnMesh.getPoints(inMeshMPointArray, om.MSpace.kWorld)
            inMeshMIntArray_vertexCount = om.MIntArray()
            inMeshMIntArray_vertexList = om.MIntArray()
            currentInMeshMFnMesh.getVertices(inMeshMIntArray_vertexCount, inMeshMIntArray_vertexList);
            for i in range(inMeshMIntArray_vertexCount.length()):
                self.vertex_count_list.append(inMeshMIntArray_vertexCount[i])
            for i in range(inMeshMIntArray_vertexList.length()):
                self.vertex_list_list.append(inMeshMIntArray_vertexList[i])
            # put each point to a list
            for i in range( inMeshMPointArray.length() ) :
                #v1 = math.floor(inMeshMPointArray[i][0] * 10000)/10000
                #v2 = math.floor(inMeshMPointArray[i][0] * 10000)/10000
                #v3 = math.floor(inMeshMPointArray[i][0] * 10000)/10000
                self.points_list.append( [inMeshMPointArray[i][0], inMeshMPointArray[i][1], inMeshMPointArray[i][2]] )
            #pointList.append( [v1, v2, v3] )
            break

class Helper(QtCore.QObject):
    def __init__(self) -> None:
        super(Helper, self).__init__()
        
        self.api = MayaApi()
        self.server_started = False
        self.my_worker = None
        self.my_thread = None
        
        self.host_address = None
        self.server_port = None
        self.host_name = socket.gethostname()
        self.ip_address = socket.gethostbyname(self.host_name)
        self.cam_trans = []

    def start_server(self):
        print("-"*10, "Start Server Thread")
        if not self.server_started:
            self.my_thread = QtCore.QThread()
            self.my_worker = ServerWorker(self.ip_address, self.server_port)
            self.my_worker.tcpSignal.connect(self.onTcpSignal)
            self.my_worker.moveToThread(self.my_thread)
            self.my_thread.started.connect(self.my_worker.run)
            self.my_worker.finished.connect(self.my_thread.quit)
            self.my_worker.finished.connect(self.my_worker.deleteLater)
            self.my_thread.finished.connect(self.my_thread.deleteLater)
            self.my_worker.finished.connect(self.onWorkerFinished)
            self.my_thread.start()
            self.server_started = True
        else:
            print("-"*10, "The Server Has Started")

    def stop_server(self):
        print("-"*10, "Stop Server")
        if self.server_started:
            self.my_worker.stop()
            print("-"*10, "Server Stopped")
        else:
            print("-"*10, "Server Not Started")
    
    def send_hello(self):
        _addr = 'http://{}/hello'.format(self.host_address)
        print("SendHello ", _addr)
        r = requests.get(_addr)
        print("SendHello ", r)

    def send_client_info(self, rem):
        _addr = 'http://{}/set_client_info'.format(self.host_address)
        print("SendClientInfo ", _addr)
        data = {"Host": self.ip_address, "Port": self.server_port, "Remove": rem}
        r = requests.post(_addr, json=data)
        print("SendClientInfo ", r)

    def route_hello(self):
        global maya_basicTest_window
        print("-"*10, " info ", "-"*10)
        print("GlobalWindow", maya_basicTest_window)
        print("-"*10, "      ", "-"*10)

    def route_set_frame(self, frame):
        global maya_basicTest_window
        print("Set Frame ", frame)
        self.api.setCurrentFrame(frame)
    
    @Slot(str)
    def onTcpSignal(self, d):
        print("onTcpSignal", d)
        ds = d.split(' ')
        if ds[0] == "FRAME":
            frame = int(ds[1])
            self.route_set_frame(frame)
        if ds[0] == "HELLO":
            self.route_hello()
    
    @Slot()
    def onWorkerFinished(self):
        self.server_started = False


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__()
        self.helper = Helper()
        self.setMinimumSize(520, 520)

        self.helloButton = QtWidgets.QPushButton()
        self.debugButton = QtWidgets.QPushButton()
        self.startServerButton = QtWidgets.QPushButton()
        self.stopServerButton = QtWidgets.QPushButton()
        self.hostInput = QtWidgets.QLineEdit()
        self.serverPort = QtWidgets.QLineEdit()
        self.hostInputLabel = QtWidgets.QLabel()
        self.serverPortLabel = QtWidgets.QLabel()
        self.serverPortInputLayout = QtWidgets.QHBoxLayout()
        self.hostInputLayout = QtWidgets.QHBoxLayout()
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.settingLayout = QtWidgets.QVBoxLayout()
        self.debugLayout = QtWidgets.QVBoxLayout()
        self.centralw = QtWidgets.QWidget()

        self.hostInputLayout.addWidget(self.hostInputLabel)
        self.hostInputLayout.addWidget(self.hostInput)
        self.serverPortInputLayout.addWidget(self.serverPortLabel)
        self.serverPortInputLayout.addWidget(self.serverPort)
        self.settingLayout.addWidget(self.helloButton)
        self.settingLayout.addWidget(self.startServerButton)
        self.settingLayout.addWidget(self.stopServerButton)
        self.debugLayout.addWidget(self.debugButton)
        self.mainLayout.addLayout(self.hostInputLayout)
        self.mainLayout.addLayout(self.serverPortInputLayout)
        self.mainLayout.addLayout(self.settingLayout)
        self.mainLayout.addLayout(self.debugLayout)

        self.centralw.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralw)

        self.menubar = QtWidgets.QMenuBar(self)
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        self.connectUi()

    def retranslateUi(self):
        host = "192.168.3.15:18080"
        port = "18081"
        self.setWindowTitle("Zeno Live Sync")
        self.helloButton.setText("hello")
        self.debugButton.setText("debug")
        self.startServerButton.setText("start server")
        self.stopServerButton.setText("stop server")
        self.hostInput.setText(host)
        self.serverPort.setText(port)
        self.hostInputLabel.setText("Host")
        self.serverPortLabel.setText("Port")

        self.helper.host_address = host
        self.helper.server_port = port
    
    def connectUi(self):
        self.helloButton.clicked.connect(self.onHelloButtonClicked)
        self.debugButton.clicked.connect(self.onDebugButtonClicked)
        self.startServerButton.clicked.connect(self.onStartServerButtonClicked)
        self.stopServerButton.clicked.connect(self.onStopServerButtonClicked)
        self.hostInput.textChanged.connect(self.onHostTextChanged)
        self.serverPort.textChanged.connect(self.onServerPortTextChanged)

    def onHostTextChanged(self):
        print("onHostTextChanged ", self.hostInput.text())
        self.helper.host_address = self.hostInput.text()

    def onServerPortTextChanged(self):
        self.helper.server_port = self.serverPort.text()

    def onDebugButtonClicked(self):
        self.helper.api.perspCameraData()
        self.helper.api.selectionMeshData()
        print("=========="*2)
        print("Global Window ", maya_basicTest_window)
        print("Camera Data ", self.helper.cam_trans)
        print("Mesh Data Points ", self.helper.api.points_list)
        print("Mesh Data Vertex ", self.helper.api.vertex_list_list)
        print("Mesh Data Counts ", self.helper.api.vertex_count_list)
        print("=========="*2)

    def onHelloButtonClicked(self):
        self.helper.send_hello()
    
    def onStartServerButtonClicked(self):
        self.helper.start_server()
        self.helper.send_client_info(False)
    
    def onStopServerButtonClicked(self):
        self.helper.stop_server()
        self.helper.send_client_info(True)

    def closeEvent(self, event):
        print("-"*10, "Window close")
        self.onStopServerButtonClicked()
        event.accept()


def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken2.wrapInstance(int(ptr), QtCore.QObject)

def mayaMain():
    global maya_basicTest_window
    try:
        maya_basicTest_window.onStopServerButtonClicked()
        maya_basicTest_window.close()
    except:
        pass
    print("-"*10, "Window show")

    maya_basicTest_window = Ui_MainWindow(getMayaWindow())
    maya_basicTest_window.show()
    maya_basicTest_window.onStartServerButtonClicked()

mayaMain()