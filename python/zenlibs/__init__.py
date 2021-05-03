import os

import zenutils


libdir = os.path.dirname(os.path.abspath(__file__))


def import_library(name):
    return zenutils.import_library(libdir, name)


def load_library(name):
    return zenutils.load_library(os.path.join(libdir, name) + '.so')
