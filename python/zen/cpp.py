'''
C++ APIs
'''

from .core import get_core


def dumpDescriptors():
    return get_core().dumpDescriptors()

def addNode(type, name):
    return get_core().addNode(type, name)

def initNode(name):
    return get_core().initNode(name)

def applyNode(name):
    return get_core().applyNode(name)

def setNodeInput(name, key, srcname):
    return get_core().setNodeInput(name, key, srcname)

def setNodeParam(name, key, value):
    return get_core().setNodeParam(name, key, value)

def getNodeRequirements(name):
    return get_core().getNodeRequirements(name)

def isCppObject(name):
    return get_core().isCppObject(name)

