import os
import runpy
import tempfile
import threading
import functools
from multiprocessing import Pool


class LazyImport:
    def __init__(self, name):
        self.__name = name
        self.__module = None

    def __getattr__(self, attr):
        if self.__module is None:
            print('* LazyImport:', self.__name)
            self.__module = __import__(self.__name)
        return getattr(self.__module, attr)


def go(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


def _our_run(param):
    func, args, kwargs = param
    return func(*args, **kwargs)


def multiprocgo(func, *args, **kwargs):
    param = func, args, kwargs
    return Pool().map(_our_run, [param])[0]


def multiproc(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return multiprocgo(func, *args, **kwargs)

    return wrapped


def goes(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return go(func, *args, **kwargs)

    return wrapped


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


def run_script(src, callback=runpy.run_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'script.py')
        with open(path, 'w') as f:
            f.write(src)
        return callback(path)
