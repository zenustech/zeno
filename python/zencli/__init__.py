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
    go(run_script, script, capture_output=False)


def getDescriptors():
    script = std_header + f'''
descs = zen.dumpDescriptors()
print('=--=', descs, '=--=')
'''
    output = run_script(script, capture_output=True)
    descs = output.split(b'=--=')[1]
    descs = descs.decode().splitlines()
    descs = [parse_descriptor_line(line) for line in descs if ':' in line]
    descs = {name: Descriptor(*args) for name, *args in descs}
    return descs
