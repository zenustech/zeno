'''
Numpy APIs
'''

from .core import get_core


def setNumpyObject(name, arr):
    return get_core().setNumpyObject(name, arr)

def isNumpyObject(name):
    return get_core().isNumpyObject(name)

class getNumpyObjectMeta:
    def __init__(self, name):
        meta = get_core().getNumpyObjectMeta(name)
        self.ptr, self.itemsize, self.format, \
            self.ndim, self.shape, self.strides, \
            self.isref = meta

def getNumpyObject(name):
    if not isNumpyObject(name):
        return name

    meta = getNumpyObjectMeta(name)

    import numpy as np
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

    return getattr(get_core(), 'getNumpyObject_' + c_type)(name)
