import ctypes, os

from .utils import rel2abs, os_name

'''
if os_name == 'win32':
    dllpath = rel2abs(__file__, 'lib')
    os.environ['PATH'] += os.pathsep + dllpath
    ctypes.cdll.LoadLibrary('zeno.dll')
elif os_name == 'darwin':
    ctypes.cdll.LoadLibrary(rel2abs(__file__, 'lib', 'libzeno.dylib'))
else:
    ctypes.cdll.LoadLibrary(rel2abs(__file__, 'lib', 'libzeno.so'))
'''

from . import pyzeno as core

def loadAutoloads():
    dir = rel2abs(__file__, 'lib')
    print('loading addons from', dir)
    if not os.path.isdir(dir):
        return

    paths = []
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.islink(path):
            continue
        if os_name == 'win32':
            if name.endswith('.dll'):
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
