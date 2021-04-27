import os
import sys
import tempfile
from multiprocessing import Pool
import runpy


def my_run_path(path):
    result = runpy.run_path(path)
    return {'descs': result.get('descs', None)}


def run_script(src):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'script.py')
        with open(path, 'w') as f:
            f.write(src)

        return Pool().map(my_run_path, [path])[0]


def inject_ld_preload(*pathes):
    for path in pathes:
        path = os.path.realpath(path)
        if os.path.isfile(path):
            break
    else:
        return

    ld_preload = os.environ.get('LD_PRELOAD', '')
    if ld_preload:
        ld_preload = path + ':' + ld_preload
    else:
        ld_preload = path
    os.environ['LD_PRELOAD'] = ld_preload
