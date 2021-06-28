'''
C++ APIs
'''

from zenutils import load_library, rel2abs, os_name

if os_name == 'linux':
    load_library(rel2abs(__file__, 'usr', 'lib', 'libzensession.so'))
elif os_name == 'win32':
    load_library(rel2abs(__file__, 'usr', 'lib', 'zensession.dll'))
else:
    raise RuntimeError(f'Unsupported OS: {os_name}')


from . import libzenpy as core


def dumpDescriptors():
    return core.dumpDescriptors()

def addNode(type, name):
    return core.addNode(type, name)

def initNode(name):
    return core.initNode(name)

def applyNode(name):
    return core.applyNode(name)

def setNodeInput(name, key, srcname):
    return core.setNodeInput(name, key, srcname)

def setNodeParam(name, key, value):
    return core.setNodeParam(name, key, value)

def getNodeRequirements(name):
    return core.getNodeRequirements(name)

def isCppObject(name):
    return core.isCppObject(name)

