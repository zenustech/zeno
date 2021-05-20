import os
import runpy
import shutil
import tempfile
import multiprocessing as mp
from zenutils import run_script, add_line_numbers

from .codegen import generate_script
from .descriptor import parse_descriptor_line


std_header = '''
import zen
zen.loadLibrary('libzenbase.so')
zen.loadLibrary('libzenvdb.so')
zen.loadLibrary('libOCTlib.so')
zen.loadLibrary('libFLIPlib.so')
'''

iopath = '/tmp/zenio'
g_proc = None


def launchGraph(graph, nframes):
    script = generate_script(graph)
    return launchScript(script, nframes)


def killProcess():
    if g_proc is None:
        print('worker process is not running')
        return
    g_proc.terminate()
    print('worker process killed')


def launchScript(script, nframes):
    shutil.rmtree(iopath, ignore_errors=True)
    os.mkdir(iopath)

    script = std_header + f'''
zen.setIOPath({iopath!r})
{script}

for frame in range({nframes}):
\tprint('FRAME:', frame)
\texecute()
print('EXITING')
'''
    print(add_line_numbers(script))
    if 1:
        global g_proc
        g_proc = mp.Process(target=run_script, args=[script], daemon=True)
        g_proc.start()
        g_proc.join()
        print('worker processed exited with', g_proc.exitcode)
        g_proc = None
    else:
        run_script(script)

def getDescriptors():
    script = std_header + f'''
descs = zen.dumpDescriptors()
'''
    descs = run_script(script)['descs']
    if isinstance(descs, bytes):
        descs = descs.decode()
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
