import os, time, win32api, ctypes

from zenutils import load_library, rel2abs, os_name

win32api.SetDllDirectory(rel2abs(__file__, 'lib'))
ctypes.cdll.LoadLibrary('zensession.dll')

from . import libzenpy as core

__all__ = ['core']