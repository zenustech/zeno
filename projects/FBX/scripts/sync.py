import maya.OpenMaya as OpenMaya
import sys	
sys.path.append(r"C:\Program Files\Autodesk\Maya2022\Python37\Lib\site-packages\pip\_vendor")
import requests


vertexCountList = []
vertexListList = []
pointList = []
cameraTranslation = []



def SendVertices():
	url = 'http://localhost:5236/ver'	
	myobj = {
		'vertices': pointList, 
		'vertexCount': vertexCountList,
		'vertexList': vertexListList
	}
	x = requests.post(url, json = myobj)


def SendCamera():
	url = 'http://localhost:5236/cam'	
	myobj = {
		'translation': cameraTranslation, 
	}
	x = requests.post(url, json = myobj)


def GetMeshSelectionData(  ):
	# get the active selection
	selection = OpenMaya.MSelectionList()
	OpenMaya.MGlobal.getActiveSelectionList( selection )
	iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
	print("Sel ", selection)
	# go througt selection
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

		print("vertexCountList ", vertexCountList)
		print("vertexListList ", vertexListList)

		# put each point to a list

		for i in range( inMeshMPointArray.length() ) :

			pointList.append( [inMeshMPointArray[i][0], inMeshMPointArray[i][1], inMeshMPointArray[i][2]] )

		print("pointList ", pointList)
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
	print("translate", translateX, translateY, translateZ)
	print("rotate", rotateX, rotateY, rotateZ)
	print("scale", scaleX, scaleY, scaleZ)
	cameraTranslation.append(translateX)
	cameraTranslation.append(translateY)
	cameraTranslation.append(-translateZ)
	cameraTranslation.append(rotateX)
	cameraTranslation.append(-rotateY)
	cameraTranslation.append(rotateZ)
	cameraTranslation.append(scaleX)
	cameraTranslation.append(scaleY)
	cameraTranslation.append(scaleZ)

GetPerspCameraData()
GetMeshSelectionData()
SendVertices()
SendCamera()