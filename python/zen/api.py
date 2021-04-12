'''
Unified APIs
'''

from .py import *
from . import cpp, py


def dumpDescriptors():
    return py.dumpDescriptors() + cpp.dumpDescriptors()

def addNode(type, name):
    if py.hasNodeType(type):
        py.addNode(type, name)
    else:
        cpp.addNode(type, name)

def applyNode(name):
    if py.hasNodeName(name):
        py.applyNode(name)
    else:
        cpp.applyNode(name)

def setNodeInput(name, key, srcname):
    if py.hasNodeName(name):
        py.setNodeInput(name, key, srcname)
    else:
        cpp.setNodeInput(name)

def setNodeParam(name, key, value):
    if py.hasNodeName(name):
        py.setNodeParam(name, key, value)
    else:
        cpp.setNodeParam(name)
