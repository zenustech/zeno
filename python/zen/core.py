'''
Core DLL singleton
'''


@eval('lambda x: x()')
def get_core():
    def import_core():
        import os
        import sys

        lib_dir = os.path.dirname(__file__)

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


def loadLibrary(path):
    import ctypes
    ctypes.cdll.LoadLibrary(path)
