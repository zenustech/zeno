import maya.OpenMaya as OpenMaya
import sys
import json
import math
import time
import socket
import threading
# sys.path.append(r"C:\Program Files\Autodesk\Maya2022\Python37\Lib\site-packages\pip\_vendor")
# import requests

vertexCountList = []
vertexListList = []
pointList = []
cameraTranslation = []


def SendFunc(content, pms = False):
	# host = socket.gethostname()
	host = "localhost"
	port = 5236

	contextSize = len(content)
	preChunkSize = int(contextSize / 60000)
	if pms:
		print("Send Length ", contextSize, " Size ", preChunkSize)
	if preChunkSize != 0:
		chunks, chunk_size = len(content), len(content)//preChunkSize
		for i in range(0, chunks, chunk_size):
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.connect((host, port))
			s.send(content[i:i+chunk_size])
			s.recv(8192)
			s.close()
	else:
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((host, port))
		s.sendall(content)
		repr(s.recv(1024))
		s.close()


def SendData(dataType):
	myobj = {}
	if dataType == 1:
		SendFunc("TYPE VERT".encode("utf-8"))
		myobj = {
			'vertices': pointList,
			'vertexCount': vertexCountList,
			'vertexList': vertexListList
		}
	if dataType == 2:
		SendFunc("TYPE CAME".encode("utf-8"))
		myobj = {
			'translation': cameraTranslation,
		}
	SendFunc(json.dumps(myobj, indent=0).encode('utf-8'), True)
	SendFunc("SEND DONE".encode("utf-8"))

def GetMeshSelectionData(  ):
	# get the active selection
	selection = OpenMaya.MSelectionList()
	OpenMaya.MGlobal.getActiveSelectionList( selection )
	iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
	#print("Sel ", selection)
	# go througt selection
	vertexCountList.clear()
	vertexListList.clear()
	pointList.clear()

	while not iterSel.isDone():

		# get dagPath
		dagPath = OpenMaya.MDagPath()
		iterSel.getDagPath( dagPath )

		# create empty point array
		inMeshMPointArray = OpenMaya.MPointArray()

		# create function set and get points in world space
		currentInMeshMFnMesh = OpenMaya.MFnMesh(dagPath)
		currentInMeshMFnMesh.getPoints(inMeshMPointArray, OpenMaya.MSpace.kWorld)

		inMeshMIntArray_vertexCount = OpenMaya.MIntArray()
		inMeshMIntArray_vertexList = OpenMaya.MIntArray()
		currentInMeshMFnMesh.getVertices(inMeshMIntArray_vertexCount, inMeshMIntArray_vertexList);

		for i in range(inMeshMIntArray_vertexCount.length()):
			vertexCountList.append(inMeshMIntArray_vertexCount[i])
		for i in range(inMeshMIntArray_vertexList.length()):
			vertexListList.append(inMeshMIntArray_vertexList[i])

		#print("vertexCountList ", vertexCountList)
		#print("vertexListList ", vertexListList)

		# put each point to a list

		for i in range( inMeshMPointArray.length() ) :

			#v1 = math.floor(inMeshMPointArray[i][0] * 10000)/10000
			#v2 = math.floor(inMeshMPointArray[i][0] * 10000)/10000
			#v3 = math.floor(inMeshMPointArray[i][0] * 10000)/10000
			pointList.append( [inMeshMPointArray[i][0], inMeshMPointArray[i][1], inMeshMPointArray[i][2]] )
		#pointList.append( [v1, v2, v3] )

		#print("pointList ", pointList)
		return pointList


def GetPerspCameraData():
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
	cameraTranslation.clear()
	cameraTranslation.append(-translateX)
	cameraTranslation.append(translateY)
	cameraTranslation.append(-translateZ)
	cameraTranslation.append(-rotateX)
	cameraTranslation.append(rotateY)
	cameraTranslation.append(-rotateZ)
	cameraTranslation.append(scaleX)
	cameraTranslation.append(scaleY)
	cameraTranslation.append(scaleZ)

def thread_function():
	count = 0;
	while True:
		GetPerspCameraData()
		SendData(2)
		GetMeshSelectionData()
		SendData(1)
		time.sleep(0.1)
		count+=1
		if count > 100:
			break

# GetPerspCameraData()
GetMeshSelectionData()
SendData(1)
# SendData(2)

#x = threading.Thread(target=thread_function, args=())
#x.start()
