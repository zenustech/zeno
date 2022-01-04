import os
import sys
import uuid
import runpy
import ctypes
import hashlib
import tempfile
import threading
import functools
import traceback
from contextlib import contextmanager
from multiprocessing import Pool


os_name = sys.platform


def rel2abs(file, *args):
    return os.path.join(os.path.dirname(os.path.abspath(file)), *args)


def treefiles(dir):
    if not os.path.isdir(dir):
        yield dir
    else:
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            yield from treefiles(path)


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


def _do_multiproc_evaluate(param):
    func, args, kwargs = param
    return func(*args, **kwargs)


def multiproc_evaluate(func, *args, **kwargs):
    param = func, args, kwargs
    return Pool().map(_do_multiproc_evaluate, [param])[0]


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


def run_script(src, callback=runpy.run_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'script.py')
        with open(path, 'w') as f:
            f.write(src)
        return callback(path)


def load_library(path, ignore_errors=False):
    if ignore_errors:
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            traceback.print_exc()
            return None
    else:
        return ctypes.cdll.LoadLibrary(path)


def import_library(libdir, name):
    assert os.path.isdir(libdir), libdir
    sys.path.insert(0, libdir)
    try:
        module = __import__(name)
    finally:
        assert sys.path.pop(0) == libdir
    return module


def add_line_numbers(script):
    res = ''
    for i, line in enumerate(script.splitlines()):
        res += '{:4d} | {}\n'.format(i + 1, line)
    return res


def gen_unique_ident(suffix):
    uid = uuid.uuid1().bytes
    uid = hashlib.md5(uid).hexdigest()[:8]
    return uid + '-' + suffix
    #return ''.join(reversed(base64.b64encode(random.randbytes(4) + struct.pack('L', time.time_ns())).decode()))
