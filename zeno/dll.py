import ctypes, os

from zenutils import rel2abs, os_name

if os_name == 'win32':
    dllpath = rel2abs(__file__, 'lib')
    os.environ['PATH'] += os.pathsep + dllpath
    ctypes.cdll.LoadLibrary('zeno.dll')
elif os_name == 'darwin':
    ctypes.cdll.LoadLibrary(rel2abs(__file__, 'lib', 'libzeno.dylib'))
else:
    ctypes.cdll.LoadLibrary(rel2abs(__file__, 'lib', 'libzeno.so'))

from . import pyzeno as core
from .addon import *

__all__ = ['core']
