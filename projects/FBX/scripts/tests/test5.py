import maya.OpenMayaUI as mui
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import maya.cmds as cmds

selection = om.MSelectionList()
om.MGlobal.getActiveSelectionList( selection )
iterSel = om.MItSelectionList(selection, om.MFn.kMesh)

while not iterSel.isDone():
    dagPath = om.MDagPath()
    iterSel.getDagPath( dagPath )
    fullPath = dagPath.fullPathName()
    fullPath = fullPath.replace("|", "_")[1:]
    print("Dag Path", dagPath)
    print("Dag Name", fullPath)

    iterSel.next()
