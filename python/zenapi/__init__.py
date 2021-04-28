import runpy
import tempfile
from zenutils import multiproc, run_script

from .codegen import generate_script
from .descriptor import parse_descriptor_line


std_header = '''
import zen
zen.loadLibrary('build/FastFLIP/libFLIPlib.so')
'''


def launchGraph(graph, nframes):
    script = generate_script(graph)
    return launchScript(script, nframes)


def _run_script(path):
    runpy.run_path(path)

def launchScript(script, nframes):
    iopath = '/tmp/zenio'
    script = std_header + f'''
zen.setIOPath({iopath!r})
{script}

for frame in range({nframes}):
\tprint('FRAME:', frame)
\texecute()
print('EXITING')
'''
    print(script)
    return run_script(script, multiproc(_run_script))


def _run_get_desc(path):
    return runpy.run_path(path)['descs']

def getDescriptors():
    script = std_header + f'''
descs = zen.dumpDescriptors()
print(descs)
'''
    descs = run_script(script, multiproc(_run_get_desc))
    if isinstance(descs, bytes):
        descs = descs.decode()
    descs = descs.splitlines()
    descs = [parse_descriptor_line(line) for line in descs if ':' in line]
    descs = {name: desc for name, desc in descs}
    return descs
