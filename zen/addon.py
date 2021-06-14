import os, ctypes

from zenutils import load_library, rel2abs, os_name

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
            if name.endswith('.dll'):
                print('Loading addon [{}]'.format(name))
                ctypes.cdll.LoadLibrary(name)

if not os.environ.get('ZEN_NOAUTOLOAD'):
    loadAutoloads()

__all__ = ['getInstallDir', 'getIncludeDir', 'getLibraryDir', 'getAutoloadDir', 'getCMakeDir', 'loadAutoloads']
