'''
Zeno Python API module
'''


import ctypes
import functools
from typing import Union, Optional, Any, Iterator, Iterable, Callable
from types import MappingProxyType

Numeric = Union[int, float, Iterable[int], Iterable[float]]
Literial = Union[int, float, Iterable[int], Iterable[float], str]


def initDLLPath(path: str):
    global api
    api = ctypes.cdll.LoadLibrary(path)

    def chkerr(ret):
        if ret != 0:
            msgRet_ = ctypes.c_char_p()
            api.Zeno_GetLastError(ctypes.pointer(msgRet_))
            raise RuntimeError('[zeno internal error] {}'.format(msgRet_.value.decode()))  # type: ignore

    def wrapchkerr(func):
        @functools.wraps(func)
        def wrapped(*args):
            chkerr(func(*args))
        return wrapped

    def define(rettype, funcname, *argtypes, do_checks=True):
        func = getattr(api, funcname)
        func.rettype = rettype
        func.argtypes = argtypes
        if do_checks:
            func = wrapchkerr(func)
            setattr(api, funcname, func)

    define(ctypes.c_uint32, 'Zeno_GetLastError', ctypes.POINTER(ctypes.c_char_p), do_checks=False)
    define(ctypes.c_uint32, 'Zeno_CreateGraph', ctypes.POINTER(ctypes.c_uint64))
    define(ctypes.c_uint32, 'Zeno_DestroyGraph', ctypes.c_uint64)
    define(ctypes.c_uint32, 'Zeno_GraphIncReference', ctypes.c_uint64)
    define(ctypes.c_uint32, 'Zeno_GraphLoadJson', ctypes.c_uint64, ctypes.c_char_p)
    define(ctypes.c_uint32, 'Zeno_GraphCallTempNode', ctypes.c_uint64, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
    define(ctypes.c_uint32, 'Zeno_GetLastTempNodeResult', ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64))
    define(ctypes.c_uint32, 'Zeno_CreateObjectInt', ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_CreateObjectFloat', ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_CreateObjectString', ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_CreateObjectPrimitive', ctypes.POINTER(ctypes.c_uint64))
    define(ctypes.c_uint32, 'Zeno_DestroyObject', ctypes.c_uint64)
    define(ctypes.c_uint32, 'Zeno_ObjectIncReference', ctypes.c_uint64)
    define(ctypes.c_uint32, 'Zeno_GetObjectLiterialType', ctypes.c_uint64, ctypes.POINTER(ctypes.c_int))
    define(ctypes.c_uint32, 'Zeno_GetObjectInt', ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_GetObjectFloat', ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_GetObjectString', ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_size_t))
    define(ctypes.c_uint32, 'Zeno_GetObjectPrimData', ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int))
# ZENO_CAPI Zeno_Error Zeno_AddObjectPrimAttr(Zeno_Object object_, Zeno_PrimMembType primArrType_, const char *attrName_, Zeno_PrimDataType dataType_) ZENO_CAPI_NOEXCEPT;
    define(ctypes.c_uint32, 'Zeno_AddObjectPrimAttr', ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    define(ctypes.c_uint32, 'Zeno_GetObjectPrimDataKeys', ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_char_p))
    define(ctypes.c_uint32, 'Zeno_ResizeObjectPrimData', ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t)
    define(ctypes.c_uint32, 'Zeno_InvokeObjectFactory', ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.py_object)
    define(ctypes.c_uint32, 'Zeno_InvokeObjectDefactory', ctypes.c_uint64, ctypes.c_char_p, ctypes.POINTER(ctypes.py_object))
    define(ctypes.c_uint32, 'Zeno_InvokeCFunctionPtr', ctypes.py_object, ctypes.c_char_p, ctypes.POINTER(ctypes.py_object))


class ZenoObject:
    _handle: int

    __create_key = object()

    def __init__(self, create_key: object, handle: int):
        assert create_key is self.__create_key, 'ZenoObject has a private constructor'
        self._handle = handle

    def __del__(self):
        api.Zeno_DestroyObject(ctypes.c_uint64(self._handle))
        self._handle = 0

    def __repr__(self) -> str:
        return '[zeno object at {}]'.format(self._handle)

    @classmethod
    def fromHandle(cls, handle: int):
        api.Zeno_ObjectIncReference(ctypes.c_uint64(handle))
        return cls(cls.__create_key, handle)

    def toHandle(self) -> int:
        return self._handle

    def asPrim(self):
        return ZenoPrimitiveObject.fromHandle(self._handle)

    @classmethod
    def fromLiterial(cls, value: Union[Literial, 'ZenoObject']) -> 'ZenoObject':
        if isinstance(value, ZenoObject):
            return value
        else:
            return cls(cls.__create_key, cls._makeLiterial(value))

    def toLiterial(self) -> Union[Literial, 'ZenoObject']:
        ret = self._fetchLiterial(self._handle)
        if ret is None:
            return self
        else:
            return ret

    @classmethod
    def fromFunc(cls, func: Callable):
        @functools.wraps(func)
        def wrappedFunc(**kwargs):
            argObjLits: dict[str, Union[Literial, 'ZenoObject']] = {k: ZenoObject.fromHandle(v).toLiterial() for k, v in kwargs.items()}  # type: ignore
            ret = func(**argObjLits)
            if ret is None:
                return {}
            elif isinstance(ret, dict):
                retObjs = {k: ZenoObject.fromLiterial(v) for k, v in ret.items()}
                wrappedFunc._wrapRetRAII = retObjs
                return {k: v.toHandle() for k, v in retObjs.items()}
            else:
                retObj = ZenoObject.fromLiterial(ret)
                wrappedFunc._wrapRetRAII = retObj
                return {'ret': retObj.toHandle()}
        return cls(cls.__create_key, cls._makeSomeObject('FunctionObject', wrappedFunc))

    def asFunc(self) -> Callable:
        fetchedHandleVal: int = self._fetchSomeObject('FunctionObject', self._handle)
        fetchedObjRAII = ZenoObject.fromHandle(fetchedHandleVal)
        def wrappedFunc(**kwargs: dict[str, Union[Literial, 'ZenoObject']]) -> _MappingProxyWrapper:
            argObjsRAII = {k: ZenoObject.fromLiterial(v) for k, v in kwargs.items()}  # type: ignore
            argHandles = {k: v.toHandle() for k, v in argObjsRAII.items()}
            fetchedHandle = fetchedObjRAII.toHandle()
            pyHandleAndKwargs_ = (fetchedHandle, argHandles)
            pyRetHandles_ = ctypes.py_object()
            api.Zeno_InvokeCFunctionPtr(ctypes.py_object(pyHandleAndKwargs_), ctypes.c_char_p('FunctionObject_call'.encode()), ctypes.pointer(pyRetHandles_))
            retHandles = pyRetHandles_.value
            assert retHandles is not None
            del argObjsRAII
            retProxy = _MappingProxyWrapper({k: ZenoObject.fromHandle(v).toLiterial() for k, v in retHandles.items()})
            return retProxy
        return wrappedFunc

    @classmethod
    def _newPrim(cls):
        return cls(cls.__create_key, cls._makePrimitive())

    @classmethod
    def _makePrimitive(cls) -> int:
        object_ = ctypes.c_uint64(0)
        api.Zeno_CreateObjectPrimitive(ctypes.pointer(object_))
        return object_.value

    @classmethod
    def _makeSomeObject(cls, typeName_: str, ffiObj_: Any) -> int:
        object_ = ctypes.c_uint64(0)
        api.Zeno_InvokeObjectFactory(ctypes.pointer(object_), ctypes.c_char_p(typeName_.encode()), ctypes.py_object(ffiObj_))
        return object_.value

    @classmethod
    def _fetchSomeObject(cls, typeName_: str, handle_: int) -> Any:
        ffiObjRet_ = ctypes.py_object()
        api.Zeno_InvokeObjectDefactory(ctypes.c_uint64(handle_), ctypes.c_char_p(typeName_.encode()), ctypes.pointer(ffiObjRet_))
        return ffiObjRet_.value

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
    def _fetchLiterial(cls, handle: int) -> Optional[Literial]:
        litType_ = ctypes.c_int(0)
        api.Zeno_GetObjectLiterialType(ctypes.c_uint64(handle), ctypes.pointer(litType_))
        ty = litType_.value
        if ty == 1:
            return cls._fetchString(handle)
        elif ty == 11:
            return cls._fetchInt(handle)
        elif 12 <= ty <= 14:
            return cls._fetchVecInt(handle, ty - 10)
        elif ty == 21:
            return cls._fetchFloat(handle)
        elif 22 <= ty <= 24:
            return cls._fetchVecFloat(handle, ty - 20)
        else:
            return None

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
    def _makeVecInt(value: Iterable[int]) -> int:
        value = tuple(value)
        n = len(value)
        assert 1 <= n <= 4
        object_ = ctypes.c_uint64(0)
        value_ = (ctypes.c_int * n)(*value)
        api.Zeno_CreateObjectInt(ctypes.pointer(object_), value_, ctypes.c_size_t(n))
        return object_.value

    @staticmethod
    def _makeVecFloat(value: Iterable[float]) -> int:
        value = tuple(value)
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
    def _fetchVecInt(handle: int, dim: int) -> Iterable[int]:
        assert 1 <= dim <= 4
        value_ = (ctypes.c_int * dim)()
        api.Zeno_GetObjectInt(ctypes.c_uint64(handle), value_, ctypes.c_size_t(dim))
        return tuple(value_)

    @staticmethod
    def _fetchVecFloat(handle: int, dim: int) -> Iterable[float]:
        assert 1 <= dim <= 4
        value_ = (ctypes.c_float * dim)()
        api.Zeno_GetObjectFloat(ctypes.c_uint64(handle), value_, ctypes.c_size_t(dim))
        return tuple(value_)

    @staticmethod
    def _fetchString(handle: int) -> str:
        strLen_ = ctypes.c_size_t(0)
        api.Zeno_GetObjectString(ctypes.c_uint64(handle), ctypes.cast(0, ctypes.POINTER(ctypes.c_char)), ctypes.pointer(strLen_))
        value_ = (ctypes.c_char * strLen_.value)()
        api.Zeno_GetObjectString(ctypes.c_uint64(handle), value_, ctypes.pointer(strLen_))
        return bytes(value_).decode()


class _MappingProxyWrapper:
    _prox: MappingProxyType[str, Union[Literial, ZenoObject]]

    def __init__(self, lut: dict[str, Union[Literial, ZenoObject]]):
        self._prox = MappingProxyType(lut)

    def __getattr__(self, key: str) -> Union[Literial, ZenoObject]:
        return self._prox[key]

    def __getitem__(self, key: str) -> Union[Literial, ZenoObject]:
        return self._prox[key]

    def __contains__(self, key: str) -> bool:
        return key in self._prox

    def to_dict(self) -> dict[str, Union[Literial, ZenoObject]]:
        return dict(self._prox)

    # def keys(self) -> Iterable[str]:
        # return self._prox.keys()

    # def values(self) -> Iterable[Union[Literial, ZenoObject]]:
        # return self._prox.values()

    # def items(self) -> Iterable[tuple[str, Union[Literial, ZenoObject]]]:
        # return self._prox.items()

    def __iter__(self) -> Iterable[str]:
        return iter(self._prox)

    def __len__(self) -> int:
        return len(self._prox)

    def __repr__(self) -> str:
        return repr(self._prox)


class ZenoPrimitiveObject(ZenoObject):
    def _getArray(self, kind: int):
        return _AttrVectorWrapper(self._handle, kind)

    def __repr__(self) -> str:
        return '[zeno primitive at {}]'.format(self._handle)

    @classmethod
    def new(cls):
        return cls._newPrim().asPrim()

    @property
    def verts(self):
        return self._getArray(0)

    @property
    def points(self):
        return self._getArray(1)

    @property
    def lines(self):
        return self._getArray(2)

    @property
    def tris(self):
        return self._getArray(3)

    @property
    def quads(self):
        return self._getArray(4)

    @property
    def polys(self):
        return self._getArray(5)

    @property
    def loops(self):
        return self._getArray(6)

    @property
    def uvs(self):
        return self._getArray(7)

    def asObject(self):
        return ZenoObject.fromHandle(self._handle)

    # Zeno | Houdini
    # poly | prim
    # loop | vert
    # vert | point
    class HoudiniStyleAccessor:
        def __init__(self, prim: 'ZenoPrimitiveObject'):
            self._polys_pos = prim.polys['pos']
            self._loops_pos = prim.loops['pos']
            self._prim = prim

        def polypoint(self, poly_id: int, vert_id: int) -> int:
            base, len = self._polys_pos[poly_id]  # type: ignore
            if vert_id < 0 or vert_id >= len:
                raise IndexError('vert_id {} out of range [0, {}) in polygon {}'.format(vert_id, len, poly_id))
            loop_id = base + vert_id
            return self._loops_pos[loop_id]  # type: ignore

        def primpoints(self, poly_id: int) -> Iterable[int]:
            base, len = self._polys_pos[poly_id]  # type: ignore
            return (self._loops_pos[base + i] for i in range(len))  # type: ignore

        def pointprims(self, vert_id: int):
            pass

        def createPrim(self, vert_ids: Iterable[int]):
            poly_base = self._prim.polys.size()
            loop_base = self._prim.loops.size()
            vert_ids = tuple(vert_ids)
            loop_len = len(vert_ids)
            self._prim.loops.resize(loop_base + loop_len)
            self._prim.polys.resize(poly_base + 1)
            self._polys_pos[poly_base] = (loop_base, loop_len)
            for i in range(loop_len):
                self._loops_pos[loop_base + i] = vert_ids[i]

        def primAddVert(self, poly_id: int, vert_id: int):
            loop_base = self._prim.loops.size()
            old_base, old_len = self._polys_pos[poly_id]  # type: ignore
            self._prim.loops.resize(loop_base + old_len + 1)  # type: ignore
            self._polys_pos[poly_id] = (loop_base, old_len)
            for i in range(old_len):  # type: ignore
                self._loops_pos[loop_base + i] = self._loops_pos[old_base + i]  # type: ignore
            self._loops_pos[loop_base + old_len] = vert_id  # type: ignore

        # def removePrim(self, poly_id: int, andloops: bool):
        #     pass

    def getHouAccessor(self):
        return self.HoudiniStyleAccessor(self)


class _MemSpanWrapper:
    _ptr: int
    _len: int
    _type: Any
    _dim: int

    def __init__(self, ptr_: int, len_: int, type_: Any, dim_: int):
        self._ptr = ptr_
        self._len = len_
        self._type = type_
        self._dim = dim_

    def __repr__(self) -> str:
        return '[zeno attribute at {} of len {} with type {} and dim {}]'.format(self._ptr, self._len, self._type, self._dim)

    def __getitem__(self, index: int) -> Numeric:
        if index < 0 or index >= self._len:
            raise IndexError('index {} out of range [0, {})'.format(index, self._len))
        base = ctypes.cast(self._ptr, ctypes.POINTER(self._type))
        return [base[index * self._dim + i] for i in range(self._dim)] if self._dim != 1 else base[index]

    def __setitem__(self, index: int, value: Numeric):
        if index < 0 or index >= self._len:
            raise IndexError('index {} out of range [0, {})'.format(index, self._len))
        base = ctypes.cast(self._ptr, ctypes.POINTER(self._type))
        if self._dim != 1:
            for i in range(self._dim):
                base[index * self._dim + i] = value[i]  # type: ignore
        else:
            base[index] = value

    def to_numpy(self, np=None):
        if np is None:
            import numpy as np
        if self._dim != 1:
            myshape = (self._len, self._dim)
        else:
            myshape = (self._len,)
        sizeinbytes = self._dim * self._len * ctypes.sizeof(self._type)
        _dtypeLut = {
            ctypes.c_float: np.float32,
            ctypes.c_int: np.int32,
        }
        dtype = _dtypeLut[self._type]
        arr = np.empty(myshape, dtype)
        arr = np.ascontiguousarray(arr)
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(arr.ctypes.data, self._ptr, sizeinbytes)
        return arr

    def from_numpy(self, arr, np=None):
        if np is None:
            import numpy as np
        if self._dim != 1:
            myshape = (self._len, self._dim)
        else:
            myshape = (self._len,)
        sizeinbytes = self._dim * self._len * ctypes.sizeof(self._type)
        if tuple(arr.shape) != myshape:
            raise ValueError('array shape mismatch {} != {}', tuple(arr.shape), myshape)
        _dtypeLut = {
            ctypes.c_float: np.float32,
            ctypes.c_int: np.int32,
        }
        dtype = _dtypeLut[self._type]
        arr = np.ascontiguousarray(arr.astype(dtype))
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(self._ptr, arr.ctypes.data, sizeinbytes)

    def to_list(self) -> list[Numeric]:
        base = ctypes.cast(self._ptr, ctypes.POINTER(self._type))
        if self._dim != 1:
            return [[base[index * self._dim + i] for i in range(self._dim)] for index in range(self._len)]
        else:
            return [base[index] for index in range(self._len)]

    def from_list(self, lst: list[Numeric]):
        if len(lst) != self._len:
            raise ValueError('list length mismatch {} != {}', len(lst), self._len)
        base = ctypes.cast(self._ptr, ctypes.POINTER(self._type))
        if self._dim != 1:
            for index, val in enumerate(lst):
                for i in range(self._dim):
                    base[index * self._dim + i] = val[i]  # type: ignore
        else:
            for index, val in enumerate(lst):
                base[index] = val

    def raw_data(self) -> tuple[int, int, Any, int]:
        return (self._ptr, self._len, self._type, self._dim)

    def dtype(self) -> tuple[type, int]:
        return (type(self._type()), self._dim)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Numeric]:
        return iter(self.to_list())


class _AttrVectorWrapper:
    _handle: int
    _kind: int

    _typeLut = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
    ]
    _dimLut = [
        3, 1, 3, 1, 2, 2, 4, 4,
    ]
    _typeUnlut: dict[tuple[type, int], int] = {
        (float, 3): 0,
        (float, 1): 1,
        (int, 3): 2,
        (int, 1): 3,
        (float, 2): 4,
        (int, 2): 5,
        (float, 4): 6,
        (int, 4): 7,
    }

    def __init__(self, handle: int, kind: int):
        self._handle = handle
        self._kind = kind

    def add_attr(self, attrName: str, dataType: tuple[type, int]):
        dataTypeInd = self._typeUnlut[dataType]
        api.Zeno_AddObjectPrimAttr(ctypes.c_uint64(self._handle), ctypes.c_int(self._kind), ctypes.c_char_p(attrName.encode()), ctypes.c_int(dataTypeInd))
        return self.attr(attrName)

    def attr(self, attrName: str):
        ptrRet_ = ctypes.c_void_p()
        lenRet_ = ctypes.c_size_t()
        typeRet_ = ctypes.c_int()
        api.Zeno_GetObjectPrimData(ctypes.c_uint64(self._handle), ctypes.c_int(self._kind), ctypes.c_char_p(attrName.encode()), ctypes.pointer(ptrRet_), ctypes.pointer(lenRet_), ctypes.pointer(typeRet_))
        return _MemSpanWrapper(ptrRet_.value, lenRet_.value, self._typeLut[typeRet_.value], self._dimLut[typeRet_.value])  # type: ignore

    def keys(self) -> list[str]:
        count_ = ctypes.c_size_t(0)
        api.Zeno_GetObjectPrimDataKeys(ctypes.c_uint64(self._handle), ctypes.c_int(self._kind), ctypes.pointer(count_), ctypes.cast(0, ctypes.POINTER(ctypes.c_char_p)))
        keys_ = (ctypes.c_char_p * count_.value)()
        api.Zeno_GetObjectPrimDataKeys(ctypes.c_uint64(self._handle), ctypes.c_int(self._kind), ctypes.pointer(count_), keys_)
        keys: list[str] = list(map(lambda x: x.decode(), keys_))
        return keys

    def values(self) -> list[_MemSpanWrapper]:
        return [self.attr(k) for k in self.keys()]

    def items(self) -> list[tuple[str, _MemSpanWrapper]]:
        return [(k, self.attr(k)) for k in self.keys()]

    def __iter__(self) -> Iterator[tuple[str, _MemSpanWrapper]]:
        return iter(self.items())

    def __repr__(self) -> str:
        return '[zeno attribute collection of primitive at {} of kind {}]'.format(self._handle, self._kind)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __getitem__(self, key: str):
        return self.attr(key)

    def __getattr__(self, key: str):
        return self.attr(key)

    def size(self) -> int:
        ptrRet_ = ctypes.c_void_p()
        lenRet_ = ctypes.c_size_t()
        typeRet_ = ctypes.c_int()
        api.Zeno_GetObjectPrimData(ctypes.c_uint64(self._handle), ctypes.c_int(self._kind), ctypes.c_char_p('pos'.encode()), ctypes.pointer(ptrRet_), ctypes.pointer(lenRet_), ctypes.pointer(typeRet_))
        return lenRet_.value

    def resize(self, newSize: int):
        api.Zeno_ResizeObjectPrimData(ctypes.c_uint64(self._handle), ctypes.c_int(self._kind), ctypes.c_size_t(newSize))


class ZenoGraph:
    _handle: int

    __create_key = object()

    def __init__(self, create_key: object, handle: int):
        assert create_key is self.__create_key, 'ZenoGraph has a private constructor'
        self._handle = handle

    def __repr__(self) -> str:
        return '[zeno graph at {}]'.format(self._handle)

    @classmethod
    def new(cls):
        graph_ = ctypes.c_uint64(0)
        api.Zeno_CreateGraph(ctypes.pointer(graph_))
        return cls(cls.__create_key, graph_.value)

    @classmethod
    def current(cls):
        handle = _currgraph
        if handle == 0:
            raise RuntimeError('no current graph')
        api.Zeno_GraphIncReference(ctypes.c_uint64(handle))
        return cls(cls.__create_key, handle)

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
_retsRAII : dict[str, Any] = {}
_currgraph : int = 0


def has_input(key: str) -> bool:
    return key in _args


def get_input(key: str) -> ZenoObject:
    if key not in _args:
        raise KeyError('invalid input key: {}'.format(key))
    return ZenoObject.fromHandle(_args[key])


def set_output(key: str, value: ZenoObject):
    _rets[key] = ZenoObject.toHandle(value)
    _retsRAII[key] = value


def get_input2(key: str) -> Union[Literial, 'ZenoObject']:
    return ZenoObject.toLiterial(get_input(key))


def set_output2(key: str, value: Union[Literial, 'ZenoObject']):
    set_output(key, ZenoObject.fromLiterial(value))


class _TempNodeWrapper:
    def __getattr__(self, key: str):
        def wrapped(**args: dict[str, Union[Literial, ZenoObject]]):
            currGraph = ZenoGraph.current()
            def fixParamKey(k):
                return k[:-1] + ':' if k.endswith('_') else k
            store_args : dict[str, ZenoObject] = {fixParamKey(k): ZenoObject.fromLiterial(v) for k, v in args.items()}  # type: ignore
            inputs : dict[str, int] = {k: ZenoObject.toHandle(v) for k, v in store_args.items()}
            outputs : dict[str, int] = currGraph.callTempNode(key, inputs)
            rets : dict[str, Union[Literial, ZenoObject]] = {k: ZenoObject.toLiterial(ZenoObject.fromHandle(v)) for k, v in outputs.items()}
            return _MappingProxyWrapper(rets)
        wrapped.__name__ = key
        wrapped.__qualname__ = __name__ + '.no.' + key
        setattr(self, key, wrapped)
        return wrapped

no = _TempNodeWrapper()


class _GetInputWrapper:
    def __getattr__(self, key: str) -> Union[Literial, ZenoObject]:
        return get_input2(key)

    def __getitem__(self, key: str) -> Union[Literial, ZenoObject]:
        return get_input2(key)

args = _GetInputWrapper()


class _SetOutputWrapper:
    def __setattr__(self, key: str, value: Union[Literial, ZenoObject]):
        return set_output2(key, value)

    def __setitem__(self, key: str, value: Union[Literial, ZenoObject]):
        return set_output2(key, value)

rets = _SetOutputWrapper()

def register_object_type(type_alias: str):
    def decorator(cls: type): 
        nonlocal type_alias
        type_alias = type_alias[0].upper() + type_alias[1:]
        def func(self):
            return cls.fromHandle(self._handle)
        setattr(ZenoObject, f'as{type_alias}' , func)
        return cls 
    return decorator
