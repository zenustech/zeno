from .zeno import *
import ctypes
from functools import reduce
from . import zeno


def init_zpc():
    global api
    api = zeno.api
    define = zeno.init_zeno_lib.define
    define(ctypes.c_uint32, 'ZS_GetObjectZsVecData', ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(
        ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int))


@register_object_type("zsVec")
class ZsSmallVec(ZenoObject):
    _typeLut = [
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_double
    ]

    def __init__(self, create_key: object, handle: int):
        super().__init__(create_key, handle)
        dims_ret_ = ctypes.c_size_t()
        ptr_ret = ctypes.c_void_p()
        dim_x_ret_ = ctypes.c_size_t()
        dim_y_ret_ = ctypes.c_size_t()
        types_ret_ = ctypes.c_int()
        api.ZS_GetObjectZsVecData(ctypes.c_uint64(self._handle), ctypes.byref(ptr_ret), ctypes.byref(dims_ret_),
                                  ctypes.byref(dim_x_ret_), ctypes.byref(dim_y_ret_), ctypes.byref(types_ret_))
        self._type = self._typeLut[types_ret_.value]
        self._ptr = ptr_ret.value
        if dims_ret_.value == 0:
            self._shape = ()
        elif dims_ret_.value == 1:
            self._shape = (dim_x_ret_.value, )
        elif dims_ret_.value == 2:
            self._shape = (dim_x_ret_.value, dim_y_ret_.value)
        else:
            raise Exception(f"unsupported dim {dims_ret_.value}")
        self._size = reduce(lambda a, b: a * b, self.shape)

    def __repr__(self) -> str:
        return '[zs small vec at {}, with size {} and type {}]'.format(self._handle, self.size, self.type)

    def asObject(self):
        return ZenoObject.fromHandle(self._handle)

    def to_numpy(self, np=None):
        if np is None:
            import numpy as np
        sizeinbytes = self.size * ctypes.sizeof(self._type)
        _dtypeLut = {
            ctypes.c_float: np.float32,
            ctypes.c_int: np.int32,
        }
        dtype = _dtypeLut[self._type]
        arr = np.empty(self.shape, dtype)
        arr = np.ascontiguousarray(arr)
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(arr.ctypes.data, self._ptr, sizeinbytes)
        return arr

    def from_numpy(self, arr, np=None):
        if np is None:
            import numpy as np
        sizeinbytes = self.size * ctypes.sizeof(self._type)
        if tuple(arr.shape) != self.shape:
            raise ValueError('array shape mismatch {} != {}',
                             tuple(arr.shape), self.shape)
        _dtypeLut = {
            ctypes.c_double: np.float64,
            ctypes.c_float: np.float32,
            ctypes.c_int: np.int32
        }
        dtype = _dtypeLut[self._type]
        arr = np.ascontiguousarray(arr.astype(dtype))
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(self._ptr, arr.ctypes.data, sizeinbytes)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def type(self):
        return self._type
