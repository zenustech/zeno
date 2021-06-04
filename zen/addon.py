import os

from zenutils import load_library, rel2abs

def getIncludeDir():
    return rel2abs(__file__, 'usr', 'include')

def getLibraryDir():
    return rel2abs(__file__, 'usr', 'lib')

def getAutoloadDir():
    return rel2abs(__file__, 'autoload')

dir = getAutoloadDir()
for name in os.listdir(dir):
    path = os.path.join(dir, name)
    #print('autoload', path)
    load_library(path)

__all__ = ['getIncludeDir', 'getLibraryDir', 'getAutoloadDir']
