'''
zeno.dll
'''

import ctypes
from typing import Optional

_zenodllfile: Optional[str] = None
_zenodllobj = None

def getDLL():
    global _zenodllfile, _zenodllobj
    if _zenodllobj is None:
        assert _zenodllfile is not None
        _zenodllobj = ctypes.cdll.LoadLibrary(_zenodllfile)
    return _zenodllobj


__all__ = [
        'getDLL',
        ]
