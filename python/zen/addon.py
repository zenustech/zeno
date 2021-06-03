import os

from zenutils import load_library, rel2abs

def getIncludeDir():
    return rel2abs(__file__, 'usr', 'include')

def getLibraryDir():
    return rel2abs(__file__, 'usr', 'lib')

__all__ = ['getIncludeDir', 'getLibraryDir']
