'''
Zeno Python API module
'''


import ctypes
import functools
from typing import Union
from typing_extensions import Self


Literial = Union[int, float, tuple[int], tuple[float], str]


def initDLLPath(path: str):
    global api
    api = ctypes.cdll.LoadLibrary(path)

    getlasterrstr = api.Zeno_GetLastErrorStr
    def chkerr(ret):
        if ret != 0:
            raise RuntimeError('(Zeno error) {}'.format(ctypes.string_at(getlasterrstr()).decode()))
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
    _handle: int

    __create_key = object()

    def __init__(self, create_key: object, handle: int):
        assert create_key is self.__create_key, 'ZenoObject has a private constructor'
        self._handle = handle

    def __del__(self):
        api.Zeno_DestroyObject(ctypes.c_uint64(self._handle))
        self._handle = 0

    @classmethod
    def fromHandle(cls, handle: int):
        api.Zeno_ObjectIncreaseReference(ctypes.c_uint64(handle))
        return cls(cls.__create_key, handle)

    def toHandle(self) -> int:
        return self._handle

    @classmethod
    def fromLiterial(cls, value: Literial) -> Self:
        return cls(cls.__create_key, cls._makeLiterial(value))

    def toLiterial(self) -> Literial:
        return self._fetchLiterial(self._handle)

    @classmethod
    def _makeLiterial(cls, value: Literial) -> int:
        if isinstance(value, int):
            return cls._makeInt(value)
        elif isinstance(value, float):
            return cls._makeFloat(value)
        elif isinstance(value, (tuple, list)):
            if not 1 <= len(value) <= 4:
                raise ValueError('expect 1 <= len(value) <= 4, got {}'.format(len(value)))
            elif not all(isinstance(x, (int, float)) for x in value):
                raise TypeError('expect all elements in value to be int or float')
            elif all(isinstance(x, int) for x in value):
                return cls._makeVecInt(value)  # type: ignore
            else:
                return cls._makeVecFloat(value)
        elif isinstance(value, str):
            return cls._makeString(value)
        else:
            raise TypeError('expect value type to be int, float, tuple of int or float, or str, got {}'.format(type(value)))

    @classmethod
    def _fetchLiterial(cls, handle: int) -> Literial:
        return cls._fetchInt(handle)  # TODO: complete switching with Zeno_GetObjectLiterialType

    @staticmethod
    def _makeInt(value: int) -> int:
        object_ = ctypes.c_uint64(0)
        value_ = ctypes.c_int(value)
        api.Zeno_CreateObjectInt(ctypes.pointer(object_), ctypes.pointer(value_), ctypes.c_size_t(1))
        return object_.value

    @staticmethod
    def _makeFloat(value: float) -> int:
        object_ = ctypes.c_uint64(0)
        value_ = ctypes.c_float(value)
        api.Zeno_CreateObjectFloat(ctypes.pointer(object_), ctypes.pointer(value_), ctypes.c_size_t(1))
        return object_.value

    @staticmethod
    def _makeVecInt(value: tuple[int]) -> int:
        n = len(value)
        assert 1 <= n <= 4
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_int * n)(*value)
        api.Zeno_CreateObjectInt(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _makeVecFloat(value: tuple[float]) -> int:
        n = len(value)
        assert 1 <= n <= 4
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_float * n)(*value)
        api.Zeno_CreateObjectFloat(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _makeString(value: str) -> int:
        arr = value.encode()
        n = len(arr)
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_char * n)(*arr)
        api.Zeno_CreateObjectString(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _fetchInt(handle: int) -> int:
        value_ = ctypes.c_int(0)
        api.Zeno_GetObjectInt(ctypes.c_uint64(handle), ctypes.pointer(value_), ctypes.c_size_t(1))
        return value_.value

    @staticmethod
    def _fetchFloat(handle: int) -> float:
        value_ = ctypes.c_float(0)
        api.Zeno_GetObjectFloat(ctypes.c_uint64(handle), ctypes.pointer(value_), ctypes.c_size_t(1))
        return value_.value

    @staticmethod
    def _fetchVecInt(handle: int, dim: int) -> tuple[int]:
        assert 1 <= dim <= 4
        value_ = (ctypes.c_int * dim)()
        api.Zeno_GetObjectVecInt(ctypes.c_uint64(handle), value_, ctypes.c_size_t(dim))
        return tuple(value_)

    @staticmethod
    def _fetchVecFloat(handle: int, dim: int) -> tuple[float]:
        assert 1 <= dim <= 4
        value_ = (ctypes.c_float * dim)()
        api.Zeno_GetObjectVecFloat(ctypes.c_uint64(handle), value_, ctypes.c_size_t(dim))
        return tuple(value_)


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


_args : dict[str, int] = {}
_rets : dict[str, int] = {}


def has_input(key: str) -> bool:
    return key in _args


def get_input(key: str) -> ZenoObject:
    if key not in _args:
        raise KeyError('invalid input key: {}'.format(key))
    return ZenoObject.fromHandle(_args[key])


def set_output(key: str, value: ZenoObject):
    _rets[key] = ZenoObject.toHandle(value)


def get_input2(key: str) -> Literial:
    return ZenoObject.toLiterial(get_input(key))


def set_output2(key: str, value: Literial):
    set_output(key, ZenoObject.fromLiterial(value))


__all__ = [
        'ZenoGraph',
        'ZenoObject',
        'has_input',
        'get_input',
        'set_output',
        'get_input2',
        'set_output2',
        ]
