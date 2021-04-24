from .procutils import run_script, go
from .codegen import generate_script
from .descriptor import parse_descriptor_line, Descriptor


std_header = '''
import zen
zen.loadLibrary('build/FastFLIP/libFLIPlib.so')
'''


def launchGraph(graph, nframes=1):
    script = generate_script(graph)
    launchScript(script, nframes)


def launchScript(script, nframes=1):
    script = std_header + f'''
{script}

for frame in range({nframes}):
\tprint('FRAME:', frame)
\texecute()
'''
    print(script)
    return go(run_script, script)


def getDescriptors():
    script = std_header + f'''
descs = zen.dumpDescriptors()
print(descs)
'''
    descs = run_script(script)['descs']
    if isinstance(descs, bytes):
        descs = descs.decode()
    descs = descs.splitlines()
    descs = [parse_descriptor_line(line) for line in descs if ':' in line]
    descs = {name: Descriptor(*args) for name, *args in descs}
    return descs
