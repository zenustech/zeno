'''
Zeno Python API module
'''

from ctypes import c_int, c_float, c_double 
from .zeno import *
from .zpc import *


def initDLLPath(path: str):
    init_zeno_lib(path)
    init_zpc()

int = c_int 
float = c_float 
double = c_double 

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
    'initDLLPath', 
    'int', 
    'float', 
    'double'
]
