'''
Unified APIs
'''

from . import cpp, py, npy
from .kwd import G


def dumpDescriptors():
    return py.dumpDescriptors() + cpp.dumpDescriptors()

def addNode(type, name):
    if py.isPyNodeType(type):
        py.addNode(type, name)
    else:
        cpp.addNode(type, name)

def touchNode(type, name):
    if py.isPyNodeType(type):
        py.touchNode(type, name)
    else:
        cpp.touchNode(type, name)

def applyNode(name):
    if py.isPyNodeName(name):
        py.applyNode(name)
    else:
        deps = cpp.getNodeRequirements(name)
        for dep in deps:
            requireObject(dep, is_py_dst=False)
        cpp.applyNode(name)

def cpp2pyObject(name):
    obj = npy.getCppObject(name)
    py.setObject(name, obj)

def py2cppObject(name):
    obj = py.getObject(name)
    npy.setCppObject(name, obj)

def setNodeInput(name, key, srcname):
    if py.isPyNodeName(name):
        py.setNodeInput(name, key, srcname)
    else:
        cpp.setNodeInput(name, key, srcname)

def setNodeParam(name, key, value):
    if py.isPyNodeName(name):
        py.setNodeParam(name, key, value)
    else:
        cpp.setNodeParam(name, key, value)

def isObject(srcname):
    return py.isPyObject(srcname) or cpp.isCppObject(srcname)

def requireObject(srcname, is_py_dst=True):
    if isObject(srcname):
        return

    nodename, sockname = srcname.split('::')
    applyNode(nodename)
    is_py_src = py.isPyNodeName(nodename)

    if is_py_src and not is_py_dst:
        if srcname in G.is_py2cpp_table or py.isPyObject(srcname):
            G.is_py2cpp_table.add(srcname)
            py2cppObject(srcname)
    if not is_py_src and is_py_dst:
        if srcname in G.is_cpp2py_table or not py.isPyObject(srcname):
            G.is_cpp2py_table.add(srcname)
            cpp2pyObject(srcname)


__all__ = [
    'addNode',
    'applyNode',
    'setNodeInput',
    'setNodeParam',
    'dumpDescriptors',
]
