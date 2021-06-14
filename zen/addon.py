import os, ctypes

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
    if os.path.isdir(dir):
        for name in os.listdir(dir):
            if os_name == 'win32':
                if name.endswith('.dll'):
                    print('Loading addon [{}]'.format(name))
                    ctypes.cdll.LoadLibrary(name)
            else:
                if name.endswith('.so'):
                    path = os.path.join(dir, name)
                    print('Loading addon [{}]'.format(path))
                    ctypes.cdll.LoadLibrary(path)

if not os.environ.get('ZEN_NOAUTOLOAD'):
    loadAutoloads()

__all__ = ['getInstallDir', 'getIncludeDir', 'getLibraryDir', 'getAutoloadDir', 'getCMakeDir', 'loadAutoloads']
