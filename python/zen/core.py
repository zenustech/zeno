'''
Core DLL singleton
'''


def get_core():
    from . import libzenpy as core
    return core
