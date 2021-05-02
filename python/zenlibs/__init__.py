import os


libdir = os.path.dirname(os.path.abspath(__file__))


def import_library(name):
    from zenutils import import_library
    return import_library(libdir, name)


def load_library(name):
    from zenutils import load_library
    return load_library(os.path.join(libdir, name) + '.so')
