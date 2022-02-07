'''
Numpy APIs
'''

import numpy as np

from .cpp import core
from .py import BooleanObject


def setArrayObject(name, arr):
    return core.setArrayObject(name, arr)

class getArrayObjectMeta:
    def __init__(self, name):
        meta = core.getArrayObjectMeta(name)
        self.ptr, self.itemsize, self.format, \
            self.ndim, self.shape, self.strides, \
            self.isref = meta

def getArrayObject(name):
    meta = getArrayObjectMeta(name)
    TYPES = {
            np.uint8: 'uint8_t',
            np.uint16: 'uint16_t',
            np.uint32: 'uint32_t',
            np.uint64: 'uint64_t',
            np.int8: 'int8_t',
            np.int16: 'int16_t',
            np.int32: 'int32_t',
            np.int64: 'int64_t',
            np.float32: 'float',
            np.float64: 'double',
    }
    np_format = np.dtype(meta.format)
    for np_type, c_type in TYPES.items():
        if np_format == np_type:
            break
    else:
        raise KeyError(f'bad numpy data format: {meta.format}')

    return getattr(core, 'getArrayObject_' + c_type)(name)


def setReference(name, srcname):
    return core.setReference(name, srcname)

def getReference(name):
    return core.getReference(name)

def setBooleanObject(name, value):
    return core.setBooleanObject(name, value)

def getBooleanObject(name):
    return core.getBooleanObject(name)

def setNumericObject(name, value):
    return core.setNumericObject(name, value)

def getNumericObject(name):
    return core.getNumericObject(name)

def setStringObject(name, value):
    return core.setStringObject(name, value)

def getStringObject(name):
    return core.getStringObject(name)

def getCppObjectType(name):
    return core.getCppObjectType(name)



class Reference:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def setCppObject(name, obj):
    if isinstance(obj, np.ndarray):
        setArrayObject(name, obj)
    elif isinstance(obj, Reference):
        setReference(name, str(obj))
    elif isinstance(obj, BooleanObject):
        setBooleanObject(name, bool(obj))
    elif isinstance(obj, str):
        setStringObject(name, obj)
    elif isinstance(obj, (int, float, list, tuple)):
        setNumericObject(name, obj)
    else:
        raise RuntimeError(f'unsupported type {type(obj)} to pass into C++')

def getCppObject(name):
    type = getCppObjectType(name)
    if type == 'array':
        return getArrayObject(name)
    if type == 'boolean':
        return BooleanObject(getBooleanObject(name))
    if type == 'string':
        return getStringObject(name)
    if type == 'numeric':
        return getNumericObject(name)
    else:
        return Reference(name)


__all__ = ['Reference']
