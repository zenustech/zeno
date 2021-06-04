import os

from zenutils import load_library, rel2abs

def getInstallDir():
    return rel2abs(__file__)

def getIncludeDir():
    return rel2abs(__file__, 'usr', 'include')

def getLibraryDir():
    return rel2abs(__file__, 'usr', 'lib')

def getAutoloadDir():
    return rel2abs(__file__, 'autoload')

dir = getAutoloadDir()
if os.path.isdir(dir):
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        load_library(path, ignore_errors=True)

__all__ = ['getInstallDir', 'getIncludeDir', 'getLibraryDir', 'getAutoloadDir']
