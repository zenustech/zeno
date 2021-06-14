import os, time

from zenutils import load_library, add_library_path, rel2abs, os_name

if os_name == 'linux':
    libpath = rel2abs(__file__, 'lib', 'libzensession.so')
    load_library(libpath)
elif os_name == 'win32':
    libname = 'zensession.dll'
    libdir = rel2abs(__file__, 'lib')
    add_library_path(libdir)
    load_library(libname)

from . import libzenpy as core

__all__ = ['core']