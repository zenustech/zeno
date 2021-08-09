import ctypes, os, sys

'''
from .utils import os_name
from ..utils import relative_path

lib_dir = relative_path('lib')

if os_name == 'win32':
    os.environ['PATH'] += os.pathsep + lib_dir
    if sys.version_info >= (3, 8):
        # for py 3.8+
        #   https://docs.python.org/3.8/whatsnew/3.8.html#ctypes
        os.add_dll_directory(lib_dir)
    ctypes.cdll.LoadLibrary('zeno.dll')
elif os_name == 'darwin':
    ctypes.cdll.LoadLibrary(os.path.join(lib_dir, 'libzeno.dylib'))
else:
    ctypes.cdll.LoadLibrary(os.path.join(lib_dir, 'libzeno.so'))
'''

from .utils import os_name
from ..utils import relative_path

if os_name == 'win32':  # windows doesn't support rpath, let's mock it only
    lib_dir = relative_path('lib')
    os.environ['PATH'] += os.pathsep + lib_dir
    if sys.version_info >= (3, 8):
        os.add_dll_directory(lib_dir)

from .. import zeno_pybind11_module as core

#'''
def loadAutoloads():
    print('loading addons from', lib_dir)
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
            if 'zeno_' in name and name.endswith('.dylib'):
                paths.append(name)
        else:
            if 'zeno_' in name and 'so' in name.split(os.extsep):
                paths.append(path)

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

if os_name == 'win32' and not os.environ.get('ZEN_NOAUTOLOAD'):
    loadAutoloads()
#'''

__all__ = ['core']
