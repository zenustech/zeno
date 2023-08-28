from .zeno import zeno_lib
from typing import Iterable
from ctypes import c_uint64
import ctypes

# ref: ze Python Library by Archibate
class ZenoObject:
    def __init__(self, handle) -> None:
        self.handle = handle

    @staticmethod
    def from_handle(handle):
        zeno_lib.call('Zeno_ObjectIncReference', c_uint64(handle)) 
        return ZenoObject(handle)

    def as_type(self, obj_type):
        # vector variant...
        # svec variant...
        return obj_type.from_handle(self.handle)

    def to_handle(self):
        return self.handle

    def __del__(self):
        zeno_lib.call('Zeno_DestroyObject', c_uint64(self.handle))

    def __repr__(self) -> str:
        return '[zeno object at {}]'.format(self.handle)
    
    @staticmethod
    def make_literal(value):
        if isinstance(value, int):
            return ZenoObject._makeInt(value)
        elif isinstance(value, float):
            return ZenoObject._makeFloat(value)
        elif isinstance(value, (tuple, list)):
            if not 1 <= len(value) <= 4:
                raise ValueError('expect 1 <= len(value) <= 4, got {}'.format(len(value)))
            elif not all(isinstance(x, (int, float)) for x in value):
                raise TypeError('expect all elements in value to be int or float')
            elif all(isinstance(x, int) for x in value):
                return ZenoObject._makeVecInt(value)  # type: ignore
            else:
                return ZenoObject._makeVecFloat(value)
        elif isinstance(value, str):
            return ZenoObject._makeString(value)
        else:
            raise TypeError('expect value type to be int, float, tuple of int or float, or str, got {}'.format(type(value)))

    @staticmethod
    def from_literal(value):
        if hasattr(value, 'to_handle'):
            return value
        else:
            return ZenoObject.make_literal(value)

    def fetch_literal(self):
        litType_ = ctypes.c_int(0)
        handle = self.handle
        zeno_lib.call('Zeno_GetObjectLiterialType', ctypes.c_uint64(handle), ctypes.pointer(litType_))
        ty = litType_.value
        if ty == 1:
            return ZenoObject._fetchString(handle)
        elif ty == 11:
            return ZenoObject._fetchInt(handle)
        elif 12 <= ty <= 14:
            return ZenoObject._fetchVecInt(handle, ty - 10)
        elif ty == 21:
            return ZenoObject._fetchFloat(handle)
        elif 22 <= ty <= 24:
            return ZenoObject._fetchVecFloat(handle, ty - 20)
        else:
            return None

    def to_literal(self):
        ret = self.fetch_literal()
        if ret is None:
            return self
        else:
            return ret
    
    @staticmethod
    def _makeInt(value: int) -> int:
        object_ = ctypes.c_uint64(0)
        value_ = ctypes.c_int(value)
        zeno_lib.call('Zeno_CreateObjectInt', ctypes.pointer(object_), ctypes.pointer(value_), ctypes.c_size_t(1))
        return object_.value

    @staticmethod
    def _makeFloat(value: float) -> int:
        object_ = ctypes.c_uint64(0)
        value_ = ctypes.c_float(value)
        zeno_lib.call('Zeno_CreateObjectFloat', ctypes.pointer(object_), ctypes.pointer(value_), ctypes.c_size_t(1))
        return object_.value

    @staticmethod
    def _makeVecInt(value: Iterable[int]) -> int:
        value = tuple(value)
        n = len(value)
        assert 1 <= n <= 4
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_int * n)(*value)
        zeno_lib.call('Zeno_CreateObjectInt', ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _makeVecFloat(value: Iterable[float]) -> int:
        value = tuple(value)
        n = len(value)
        assert 1 <= n <= 4
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_float * n)(*value)
        zeno_lib.call('Zeno_CreateObjectFloat', ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _makeString(value: str) -> int:
        arr = value.encode()
        n = len(arr)
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_char * n)(*arr)
        zeno_lib.call('Zeno_CreateObjectString', ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _fetchInt(handle: int) -> int:
        value_ = ctypes.c_int(0)
        zeno_lib.call('Zeno_GetObjectInt', ctypes.c_uint64(handle), ctypes.pointer(value_), ctypes.c_size_t(1))
        return value_.value

    @staticmethod
    def _fetchFloat(handle: int) -> float:
        value_ = ctypes.c_float(0)
        zeno_lib.call('Zeno_GetObjectFloat', ctypes.c_uint64(handle), ctypes.pointer(value_), ctypes.c_size_t(1))
        return value_.value

    @staticmethod
    def _fetchVecInt(handle: int, dim: int) -> Iterable[int]:
        assert 1 <= dim <= 4
        value_ = (ctypes.c_int * dim)()
        zeno_lib.call('Zeno_GetObjectInt', ctypes.c_uint64(handle), value_, ctypes.c_size_t(dim))
        return tuple(value_)

    @staticmethod
    def _fetchVecFloat(handle: int, dim: int) -> Iterable[float]:
        assert 1 <= dim <= 4
        value_ = (ctypes.c_float * dim)()
        zeno_lib.call('Zeno_GetObjectFloat', ctypes.c_uint64(handle), value_, ctypes.c_size_t(dim))
        return tuple(value_)

    @staticmethod
    def _fetchString(handle: int) -> str:
        strLen_ = ctypes.c_size_t(0)
        zeno_lib.call('Zeno_GetObjectString', ctypes.c_uint64(handle), ctypes.cast(0, ctypes.POINTER(ctypes.c_char)), ctypes.pointer(strLen_))
        value_ = (ctypes.c_char * strLen_.value)()
        zeno_lib.call('Zeno_GetObjectString', ctypes.c_uint64(handle), value_, ctypes.pointer(strLen_))
        return bytes(value_).decode()
    