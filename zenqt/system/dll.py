import ctypes, os, sys

from .utils import rel2abs, os_name

lib_dir = rel2abs(__file__, '..', 'lib')

#'''
if os_name == 'win32':
    if sys.version_info < (3, 8):
        os.environ['PATH'] += os.pathsep + lib_dir
    else:
        # for py 3.8+
        #   https://docs.python.org/3.8/whatsnew/3.8.html#ctypes
        os.add_dll_directory(lib_dir)
    ctypes.cdll.LoadLibrary('zeno.dll')
elif os_name == 'darwin':
    ctypes.cdll.LoadLibrary(os.path.join(lib_dir, 'libzeno.dylib'))
else:
    ctypes.cdll.LoadLibrary(os.path.join(lib_dir, 'libzeno.so'))
#'''

from .. import zeno_pybind11_module as core

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
            if name.endswith('.dll'):
                paths.append(name)
        elif os_name == 'darwin':
            if name.endswith('.dylib'):
                paths.append(name)
        else:
            if 'so' in name.split(os.extsep):
                paths.append(path)

    retries = {}
    max_retries = len(paths) + 2
    while paths:
        for path in list(paths):
            try:
                #print('[      ] [{}]'.format(path))
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
    loadAutoloads()

__all__ = ['core']
