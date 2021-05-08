'''
Unified APIs
'''

from . import cpp, py, npy


def dumpDescriptors():
    return py.dumpDescriptors() + cpp.dumpDescriptors()

def addNode(type, name):
    if py.isPyNodeType(type):
        py.addNode(type, name)
    else:
        cpp.addNode(type, name)

def applyNode(name):
    if py.isPyNodeName(name):
        py.applyNode(name)
    else:
        cpp.applyNode(name)

def cpp2pyObject(name):
    obj = npy.getCppObject(name)
    py.setObject(name, obj)

def py2cppObject(name):
    obj = py.getObject(name)
    npy.setCppObject(name, obj)

is_cpp2py_table = set()
is_py2cpp_table = set()

def setNodeInput(name, key, srcname):
    if py.isPyNodeName(name):
        if srcname in is_cpp2py_table or not py.isPyObject(srcname):
            is_cpp2py_table.add(srcname)
            cpp2pyObject(srcname)
        py.setNodeInput(name, key, srcname)
    else:
        if srcname in is_py2cpp_table or py.isPyObject(srcname):
            is_py2cpp_table.add(srcname)
            py2cppObject(srcname)
        cpp.setNodeInput(name, key, srcname)

def setNodeParam(name, key, value):
    if py.isPyNodeName(name):
        py.setNodeParam(name, key, value)
    else:
        cpp.setNodeParam(name, key, value)


__all__ = [
    'addNode',
    'applyNode',
    'setNodeInput',
    'setNodeParam',
    'dumpDescriptors',
]
