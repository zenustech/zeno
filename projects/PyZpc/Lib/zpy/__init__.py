import sys
from typing import Any
from .types import ZenoObject
from functools import partial
from .zeno import init_zeno_lib

# ref: ze lib by Archibate
zeno_lib_path = None
_args: dict[str, int] = {}
_rets: dict[str, int] = {}
_retsRAII: dict[str, Any] = {}
print = partial(print, file=sys.stderr)


def get_input(key):
    if key not in _args:
        raise KeyError(f'invalid input key: {key}, _args: {_args}')
    return ZenoObject.from_handle(_args[key])


def set_output(key: str, value: ZenoObject):
    _rets[key] = value.to_handle()
    _retsRAII[key] = value


def get_input2(key: str):
    return ZenoObject.to_literal(get_input(key))


def set_output2(key: str, value):
    set_output(key, ZenoObject.from_literal(value))


class _GetInputWrapper:
    def __getattr__(self, key: str):
        return get_input2(key)

    def __getitem__(self, key: str):
        return get_input2(key)


args = _GetInputWrapper()


class _SetOutputWrapper:
    def __setattr__(self, key: str, value):
        return set_output2(key, value)

    def __setitem__(self, key: str, value):
        return set_output2(key, value)


rets = _SetOutputWrapper()
