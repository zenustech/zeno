import runpy
import tempfile
import multiprocessing as mp

from .runner import run_graph, get_descriptors
from .descriptor import parse_descriptor_line


std_header = '''
'''

iopath = '/tmp/zenio'
g_proc = None


def killProcess():
    if g_proc is None:
        print('worker process is not running')
        return
    g_proc.terminate()
    print('worker process killed')


def _launch_mproc(func, *args):
    if 1:
        global g_proc
        g_proc = mp.Process(target=func, args=tuple(args), daemon=True)
        g_proc.start()
        g_proc.join()
        print('worker processed exited with', g_proc.exitcode)
        g_proc = None
    else:
        func(*args)


def launchGraph(graph, nframes):
    _launch_mproc(run_graph, graph, nframes, iopath)


def getDescriptors():
    descs = get_descriptors()
    descs = descs.splitlines()
    descs = [parse_descriptor_line(line) for line in descs if ':' in line]
    descs = {name: desc for name, desc in descs}
    print('loaded', len(descs), 'descriptors')
    return descs


__all__ = [
    'getDescriptors',
    'launchGraph',
    'killProcess',
]
