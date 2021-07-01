import os, ctypes, traceback

from zenutils import rel2abs, os_name

def getInstallDir():
    return rel2abs(__file__)

def getIncludeDir():
    return rel2abs(__file__, 'include')

def getLibraryDir():
    return rel2abs(__file__, 'lib')

def getCMakeDir():
    return rel2abs(__file__, 'cmake')

def getAutoloadDir():
    return rel2abs(__file__, 'lib')

def loadAutoloads():
    dir = getAutoloadDir()
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

__all__ = ['getInstallDir', 'getIncludeDir', 'getLibraryDir', 'getAutoloadDir', 'getCMakeDir', 'loadAutoloads']
