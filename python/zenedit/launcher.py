from collections import namedtuple

from .procutils import run_script
from .codegen import generate_script


def parse_descriptor_line(line):
    z_name, rest = line.strip().split(':', maxsplit=1)
    assert rest.startswith('(') and rest.endswith(')'), (n_name, rest)
    inputs, outputs, params, categories = rest.strip('()').split(')(')

    z_inputs = [name for name in inputs.split(',') if name]
    z_outputs = [name for name in outputs.split(',') if name]
    z_categories = [name for name in categories.split(',') if name]

    z_params = []
    for param in params.split(','):
        if not param:
            continue
        type, name, defl = param.split(':')
        z_params.append((type, name, defl))

    return z_name, z_inputs, z_outputs, z_params, z_categories


class Descriptor(namedtuple('Descriptor',
    'inputs, outputs, params, categories')):
    pass


class ZenLauncher:
    def __init__(self):
        self.header = '''
import zen
'''

    def launchGraph(self, graph, nframes=1):
        script = generate_script(graph)
        self.launchScript(script)

    def launchScript(self, script, nframes=1):
        script = self.header + f'''
{script}
for frame in range({nframes}):
    print('FRAME:', frame)
    execute()
'''
        run_script(script, capture_output=False)

    def getDescriptors(self):
        script = self.header + f'''
descs = zen.dumpDescriptors()
print('=--=', descs, '=--=')
'''
        output = run_script(script, capture_output=True)
        descs = output.split(b'=--=')[1]
        descs = descs.decode().splitlines()
        descs = [parse_descriptor_line(line) for line in descs if ':' in line]
        descs = {name: Descriptor(*args) for name, *args in descs}
        return descs
