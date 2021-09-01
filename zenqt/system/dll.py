import ctypes, os, sys, traceback

from .utils import os_name
from ..utils import relative_path

#'''
if os_name == 'win32':  # windows doesn't support rpath, let's mock him only
    lib_dir = relative_path('bin')
    os.environ['PATH'] += os.pathsep + lib_dir
    if sys.version_info >= (3, 8):
        os.add_dll_directory(lib_dir)
    del lib_dir
#'''

from ..bin import pylib_zeno as core

#'''
def loadAutoloads(lib_dir):
    #print('loading addons from', lib_dir)
    if not os.path.isdir(lib_dir):
        return

    paths = []
    for name in os.listdir(lib_dir):
        path = os.path.join(lib_dir, name)
        if os.path.islink(path):
            continue
        if os_name == 'win32':
            if name.startswith('zeno_') and name.endswith('.dll'):
                paths.append(name)
        elif os_name == 'darwin':
            if name.startswith('libzeno_') and name.endswith('.dylib'):
                paths.append(name)
        else:
            if name.startswith('libzeno_') and name.endswith('.so'):
                paths.append(path)

    #print('to be loaded:', paths)

    retries = {}
    max_retries = len(paths) + 2
    while paths:
        for path in list(paths):
            try:
                print('[      ] [{}]'.format(path))
                ctypes.cdll.LoadLibrary(path)
                paths.remove(path)
            except OSError:
                n = retries.setdefault(path, 0)
                if retries[path] > max_retries:
                    print('[FAILED] [{}]'.format(path))
                    traceback.print_exc()
                    paths.remove(path)
                else:
                    retries[path] = n + 1
            else:
                print('[  OK  ] [{}]'.format(path))

if not os.environ.get('ZEN_NOAUTOLOAD'):
    loadAutoloads(relative_path('bin'))
    loadAutoloads(relative_path('..'))
#'''

__all__ = ['core']
