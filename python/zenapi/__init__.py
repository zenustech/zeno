from .procutils import run_script
from .codegen import generate_script
from .descriptor import parse_descriptor_line


std_header = '''
import zen
zen.loadLibrary('build/FastFLIP/libFLIPlib.so')
'''


def launchGraph(graph, nframes):
    script = generate_script(graph)
    return launchScript(script, nframes)


def launchScript(script, nframes):
    script = std_header + f'''
{script}

for frame in range({nframes}):
\tprint('FRAME:', frame)
\texecute()
print('EXITING')
'''
    print(script)
    return run_script(script)


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
    descs = {name: desc for name, desc in descs}
    return descs
