import os
import sys
import runpy
import ctypes
import tempfile
import threading
import functools
from contextlib import contextmanager
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


def path_list_insert(path, pathes, append=False):
    if pathes:
        if append:
            pathes = pathes + ':' + path
        else:
            pathes = path + ':' + pathes
    else:
        pathes = path
    return path

def path_list_remove(path, pathes):
    if pathes:
        plist = pathes.split(':')
        plist.remove(path)
        ':'.join(pathes)
        return pathes
    else:
        raise ValueError('path list empty')


@contextmanager
def mock_ld_library_path(path):
    has_pathes = 'LD_LIBRARY_PATH' in os.environ
    pathes = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = path_list_insert(path, pathes)
    try:
        yield
    finally:
        if has_pathes:
            os.environ['LD_LIBRARY_PATH'] = pathes
        else:
            del os.environ['LD_LIBRARY_PATH']


def inject_ld_preload(*pathes):
    for path in pathes:
        path = os.path.realpath(path)
        if os.path.isfile(path):
            break
    else:
        return

    ld_preload = os.environ.get('LD_PRELOAD', '')
    os.environ['LD_PRELOAD'] = path_list_insert(path, ld_preload)


def run_script(src, callback=runpy.run_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'script.py')
        with open(path, 'w') as f:
            f.write(src)
        return callback(path)


def load_library(path):
    return ctypes.cdll.LoadLibrary(path)


def import_library(libdir, name):
    assert os.path.isdir(libdir), libdir
    sys.path.insert(0, libdir)
    try:
        module = __import__(name)
    finally:
        assert sys.path.pop(0) == libdir
    return module
