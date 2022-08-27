'''
zeno.dll
'''

import ctypes

core = None

def initializeDLLPath(path):
    global core
    core = ctypes.cdll.LoadLibrary(path)

    def define(func, rettype, *argtypes):
        func.rettype = rettype
        func.argtypes = argtypes

    define(ctypes.c_uint32, core.Zeno_CreateGraph, ctypes.c_void_p)
    define(ctypes.c_uint32, core.Zeno_DestroyGraph, ctypes.c_uint64)
    define(ctypes.c_uint32, core.Zeno_GraphLoadJson, ctypes.c_uint64, ctypes.c_char_p)
    define(ctypes.c_uint32, core.Zeno_GraphCallTempNode, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    define(ctypes.c_uint32, core.Zeno_CreateObjectInt, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    define(ctypes.c_uint32, core.Zeno_CreateObjectFloat, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    define(ctypes.c_uint32, core.Zeno_CreateObjectString, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    define(ctypes.c_uint32, core.Zeno_GetObjectInt, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t)
    define(ctypes.c_uint32, core.Zeno_GetObjectFloat, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t)
    define(ctypes.c_uint32, core.Zeno_GetObjectString, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)


__all__ = [
        'core',
        ]
