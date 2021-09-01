import os, sys

from .system.utils import rel2abs

def is_portable_mode():
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def relative_path(*args):
    if is_portable_mode():
        return rel2abs(sys.executable, 'zenqt', *args)
    else:
        return rel2abs(__file__, *args)

def get_executable():
    if is_portable_mode():
        return [sys.executable]
    else:
        return [sys.executable, '-m', 'zenqt.system']
