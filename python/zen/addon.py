'''
import os

from zenutils import load_library, rel2abs

for name in os.listdir(rel2abs(__file__, 'mods')):
    if not name.endswith('.so'): continue
    print('loading addon:', name)
    load_library(name)

def getLibraryPath():
    return rel2abs(__file__, 'libzen.so')

def getAddonDir():
    return rel2abs(__file__, 'mods')

def getIncludeDir():
    return rel2abs(__file__, 'include')

__all__ = ['getLibraryPath', 'getAddonDir', 'getIncludeDir']
'''
