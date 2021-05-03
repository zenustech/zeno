import os

import zenutils


this_dir = os.path.dirname(os.path.abspath(__file__))
pyd_lib_dir = os.path.join(this_dir, 'pydlib')


def import_library(name):
    return zenutils.import_library(pyd_lib_dir, name)
