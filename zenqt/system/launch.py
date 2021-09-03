import runpy
import tempfile
import threading
import atexit
import shutil
import subprocess
import json
import sys
import os
from multiprocessing import Process
from ..utils import get_executable


g_proc = None
g_iopath = None


def killProcess():
    global g_proc
    if g_proc is None:
        print('worker process is not running')
        return
    g_proc.terminate()
    #g_proc.kill()
    g_proc = None
    print('worker process killed')


@atexit.register
def cleanIOPath():
    global g_iopath
    if g_iopath is not None:
        iopath = g_iopath
        g_iopath = None
        shutil.rmtree(iopath, ignore_errors=True)


def _launch_mproc(func, *args):
    global g_proc
    if g_proc is not None:
        killProcess()
    if os.environ.get('ZEN_SPROC'):
        func(*args)
    else:
        g_proc = Process(target=func, args=tuple(args), daemon=True)
        g_proc.start()
        g_proc.join()
        if g_proc is not None:
            print('worker processed exited with', g_proc.exitcode)
        g_proc = None


def launchProgram(prog, nframes):
    global g_iopath
    global g_proc
    killProcess()
    cleanIOPath()
    g_iopath = tempfile.mkdtemp(prefix='zenvis-')
    print('IOPath:', g_iopath)
    if os.environ.get('ZEN_SPROC') or os.environ.get('ZEN_DOFORK'):
        from . import run
        _launch_mproc(run.runScene, prog['graph'], nframes, g_iopath)
    else:
        filepath = os.path.join(g_iopath, 'prog.zsg')
        with open(filepath, 'w') as f:
            json.dump(prog, f)
        # TODO: replace with binary executable
        g_proc = subprocess.Popen(get_executable() + [filepath, str(nframes), g_iopath])
        retcode = g_proc.wait()
        if retcode != 0:
            print('zeno program exited with error code:', retcode)


def getDescriptors():
    if os.environ.get('ZEN_DOFORK'):
        from . import run
        descs = run.dumpDescriptors()
    else:
        descs = subprocess.check_output(get_executable() + ['--dump-descs'])
        descs = descs.split(b'==<DESCS>==')[1].decode()
    descs = descs.splitlines()
    descs = [parse_descriptor_line(line) for line in descs
            if line.startswith('DESC@')]
    descs = {name: desc for name, desc in descs}
    print('Loaded', len(descs), 'descriptors')
    return descs


def parse_descriptor_line(line):
    _, z_name, rest = line.strip().split('@', maxsplit=2)
    assert rest.startswith('{') and rest.endswith('}'), (z_name, rest)
    inputs, outputs, params, categories = rest[1:-1].split('}{')

    z_categories = [name for name in categories.split('%') if name]

    z_inputs = []
    for input in inputs.split('%'):
        if not input:
            continue
        type, name, defl = input.split('@')
        z_inputs.append((type, name, defl))

    z_outputs = []
    for output in outputs.split('%'):
        if not output:
            continue
        type, name, defl = output.split('@')
        z_outputs.append((type, name, defl))

    z_params = []
    for param in params.split('%'):
        if not param:
            continue
        type, name, defl = param.split('@')
        z_params.append((type, name, defl))

    z_desc = {
        'inputs': z_inputs,
        'outputs': z_outputs,
        'params': z_params,
        'categories': z_categories,
    }

    return z_name, z_desc


__all__ = [
    'getDescriptors',
    'launchProgram',
    'killProcess',
]
