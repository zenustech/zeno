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


## Public APIs:

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

def initialize():
    return get_core().initialize()

def finalize():
    return get_core().finalize()

def int3(x, y, z):
    return (int(x), int(y), int(z))

def float3(x, y, z):
    return (float(x), float(y), float(z))
