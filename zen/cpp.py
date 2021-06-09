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


from . import libzenpy as _core


def dumpDescriptors():
    return _core.dumpDescriptors()

def addNode(type, name):
    return _core.addNode(type, name)

def initNode(name):
    return _core.initNode(name)

def applyNode(name):
    return _core.applyNode(name)

def setNodeInput(name, key, srcname):
    return _core.setNodeInput(name, key, srcname)

def setNodeParam(name, key, value):
    return _core.setNodeParam(name, key, value)

def getNodeRequirements(name):
    return _core.getNodeRequirements(name)

def isCppObject(name):
    return _core.isCppObject(name)

