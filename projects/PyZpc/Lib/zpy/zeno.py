import os
from .utils import CLib
import ctypes

zeno_lib = CLib()


def init_zeno_lib(path: str):
    global zeno_lib
    # ref: ze lib by Archibate
    if path is not None:
        zeno_lib.init_lib(path)

        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetLastError',
                          ctypes.POINTER(ctypes.c_char_p))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_CreateGraph',
                          ctypes.POINTER(ctypes.c_uint64))
        zeno_lib.register(
            ctypes.c_uint32, 'Zeno_DestroyGraph', ctypes.c_uint64)
        zeno_lib.register(
            ctypes.c_uint32, 'Zeno_GraphIncReference', ctypes.c_uint64)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GraphLoadJson',
                          ctypes.c_uint64, ctypes.c_char_p)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GraphCallTempNode', ctypes.c_uint64, ctypes.c_char_p, ctypes.POINTER(
            ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetLastTempNodeResult', ctypes.POINTER(
            ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_CreateObjectInt', ctypes.POINTER(
            ctypes.c_uint64), ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_CreateObjectFloat', ctypes.POINTER(
            ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_CreateObjectString', ctypes.POINTER(
            ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_CreateObjectPrimitive',
                          ctypes.POINTER(ctypes.c_uint64))
        zeno_lib.register(
            ctypes.c_uint32, 'Zeno_DestroyObject', ctypes.c_uint64)
        zeno_lib.register(
            ctypes.c_uint32, 'Zeno_ObjectIncReference', ctypes.c_uint64)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetObjectLiterialType',
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_int))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetObjectInt',
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetObjectFloat',
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetObjectString', ctypes.c_uint64,
                          ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_size_t))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetObjectPrimData', ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p,
                          ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int))

        zeno_lib.register(ctypes.c_uint32, 'Zeno_AddObjectPrimAttr',
                          ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_GetObjectPrimDataKeys', ctypes.c_uint64,
                          ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_char_p))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_ResizeObjectPrimData',
                          ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_InvokeObjectFactory', ctypes.POINTER(
            ctypes.c_uint64), ctypes.c_char_p, ctypes.py_object)
        zeno_lib.register(ctypes.c_uint32, 'Zeno_InvokeObjectDefactory',
                          ctypes.c_uint64, ctypes.c_char_p, ctypes.POINTER(ctypes.py_object))
        zeno_lib.register(ctypes.c_uint32, 'Zeno_InvokeCFunctionPtr',
                          ctypes.py_object, ctypes.c_char_p, ctypes.POINTER(ctypes.py_object))
