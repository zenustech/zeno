## Core DLL singleton

@eval('lambda x: x()')
def get_core():
    def import_core():
        import os
        import sys

        lib_dir = os.path.dirname(__file__)
        print(f'[Zen] importing core DLL from [{lib_dir}]...')

        assert os.path.exists(lib_dir)
        assert os.path.exists(os.path.join(lib_dir, 'libzenpy.so'))

        sys.path.insert(0, lib_dir)
        try:
            import libzenpy as core
        finally:
            assert sys.path.pop(0) == lib_dir
        return core

    core = None
    def get_core():
        nonlocal core
        if core is None:
            core = import_core()
        return core

    return get_core


## Load extension DLLs:

def loadLibrary(path):
    import ctypes
    print(f'[Zen] loading extension DLL at [{path}]...')
    ctypes.cdll.LoadLibrary(path)


## C++ APIs:

def dumpDescriptors():
    return get_core().dumpDescriptors()

def addNode(type, name):
    return get_core().addNode(type, name)

def applyNode(name):
    return get_core().applyNode(name)

def setNodeInput(name, key, srcname):
    return get_core().setNodeInput(name, key, srcname)

def setNodeParam(name, key, value):
    return get_core().setNodeParam(name, key, value)


## Numpy APIs:

def setNumpyObject(name, arr):
    return get_core().setNumpyObject(name, arr)

def getNumpyObjectMeta(name):
    class NumpyObjectMeta:
        pass

    meta = NumpyObjectMeta()
    meta.ptr, meta.itemsize, meta.format, meta.ndim, meta.shape, meta.strides \
            = get_core().getNumpyObjectMeta(name)
    return meta

def getNumpyObject(name):
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


## Utilities:

def int3(x, y, z):
    return (int(x), int(y), int(z))


def float3(x, y, z):
    return (float(x), float(y), float(z))
