'''
Zeno Python API module
'''


from .zeno import *
from .zpc import *


def initDLLPath(path: str):
    init_zeno_lib(path)
    init_zpc()


__all__ = [
    'ZenoGraph',
    'ZenoObject',
    'ZenoPrimitiveObject',
    'ZsSmallVec',
    'has_input',
    'get_input',
    'set_output',
    'get_input2',
    'set_output2',
    'no',
    'args',
    'rets',
    "initDLLPath"
]
