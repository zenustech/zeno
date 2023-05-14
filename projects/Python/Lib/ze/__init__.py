'''
Zeno Python API module
'''


from .zeno import * 
from .zpc import * 

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
        # '_args', 
        # '_rets', 
        # '_retsRAII', 
        # '_currgraph'
        ]
