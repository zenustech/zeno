'''
Zeno Python API module
'''


import ctypes
import functools


def initDLLPath(path: str):
    global api
    api = ctypes.cdll.LoadLibrary(path)

    getlasterrstr = api.Zeno_GetLastErrorStr
    def chkerr(ret):
        if ret != 0:
            raise RuntimeError('(Zeno error) ' + ctypes.string_at(getlasterrstr()).decode())
            # raise RuntimeError(getlasterrstr())

    def wrapchkerr(func):
        @functools.wraps(func)
        def wrapped(*args):
            chkerr(func(*args))
        return wrapped

    def define(rettype, funcname, *argtypes):
        func = getattr(api, funcname)
        func.rettype = rettype
        func.argtypes = argtypes
        func = wrapchkerr(func)
        setattr(api, funcname, func)

    define(ctypes.c_uint32, 'Zeno_GetLastErrorCode')
    define(ctypes.c_char_p, 'Zeno_GetLastErrorStr')
    define(ctypes.c_uint32, 'Zeno_CreateGraph', ctypes.POINTER(ctypes.c_uint64))
    define(ctypes.c_uint32, 'Zeno_DestroyGraph', ctypes.c_uint64)
    define(ctypes.c_uint32, 'Zeno_GraphLoadJson', ctypes.c_uint64, ctypes.c_char_p)
    define(ctypes.c_uint32, 'Zeno_GraphCallTempNode', ctypes.c_uint64, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
    define(ctypes.c_uint32, 'Zeno_GetLastTempNodeResult', ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64))
    define(ctypes.c_uint32, 'Zeno_CreateObjectInt', ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_CreateObjectFloat', ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_CreateObjectString', ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_GetObjectInt', ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_GetObjectFloat', ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_GetObjectString', ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_size_t))
    # define(ctypes.c_uint32, 'Zeno_GetObjectLiterialType', ctypes.c_uint64, ctypes.POINTER(ctypes.c_int))


class ZenoObject:
    @staticmethod
    def makeInt(value: int) -> int:
        object_ = ctypes.c_uint64(0)
        value_ = ctypes.c_int(value)
        api.Zeno_CreateObjectInt(ctypes.pointer(object_), ctypes.pointer(value_), ctypes.c_size_t(1))
        return object_.value

    @staticmethod
    def makeFloat(value: float) -> int:
        object_ = ctypes.c_uint64(0)
        value_ = ctypes.c_float(value)
        api.Zeno_CreateObjectFloat(ctypes.pointer(object_), ctypes.pointer(value_), ctypes.c_size_t(1))
        return object_.value

    @staticmethod
    def makeVecInt(value: list[int]) -> int:
        n = len(value)
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_int * n)(*value)
        api.Zeno_CreateObjectInt(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def makeVecFloat(value: list[float]) -> int:
        n = len(value)
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_float * n)(*value)
        api.Zeno_CreateObjectFloat(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def makeString(value: str) -> int:
        arr = value.encode()
        n = len(arr)
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_char * n)(*arr)
        api.Zeno_CreateObjectString(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value


class ZenoGraph:
    _handle: int

    def __init__(self):
        graph_ = ctypes.c_uint64(0)
        api.Zeno_CreateGraph(ctypes.pointer(graph_))
        self._handle = graph_.value

    def callTempNode(self, nodeType: str, inputs: dict[str, int]) -> dict[str, int]:
        inputCount_ = len(inputs)
        inputKeys_ = (ctypes.c_char_p * inputCount_)(*map(lambda x: x.encode(), inputs.keys()))
        inputObjects_ = (ctypes.c_uint64 * inputCount_)(*inputs.values())
        outputCount_ = ctypes.c_size_t(0)
        api.Zeno_GraphCallTempNode(ctypes.c_uint64(self._handle), ctypes.c_char_p(nodeType.encode()), inputKeys_, inputObjects_, ctypes.c_size_t(inputCount_), ctypes.pointer(outputCount_))
        outputKeys_ = (ctypes.c_char_p * outputCount_.value)()
        outputObjects_ = (ctypes.c_uint64 * outputCount_.value)()
        api.Zeno_GetLastTempNodeResult(outputKeys_, outputObjects_)
        outputs: dict[str, int] = dict(zip(map(lambda x: x.decode(), outputKeys_), outputObjects_))
        return outputs

    def __del__(self):
        api.Zeno_DestroyGraph(ctypes.c_uint64(self._handle))
        self._handle = 0


__all__ = [
        'ZenoGraph',
        'ZenoObject',
        ]
