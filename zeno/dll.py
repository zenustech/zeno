import ctypes

from zenutils import rel2abs, os_name

if os_name == 'win32':
    os.add_dll_directory(rel2abs(__file__, 'lib'))
    ctypes.cdll.LoadLibrary('zeno.dll')
else:
    ctypes.cdll.LoadLibrary(rel2abs(__file__, 'lib', 'libzeno.so'))

from . import pyzeno as core

__all__ = ['core']
