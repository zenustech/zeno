from collections import namedtuple

from .desc import parse_descriptor_line
from .run import run_script


class Descriptor(namedtuple('Descriptor',
    'inputs, outputs, params, categories')):
    pass


class ExecutionContext:
    def __init__(self):
        self.header = '''
import zen
'''

    def launch_script(self, script):
        script = self.header + f'''
{script}
for frame in range({nframes}):
    print('FRAME:', frame)
    execute(frame)
'''
        run_script(script, capture_output=False)

    def get_descriptors(self):
        script = self.header + f'''
descs = zen.dumpDescriptors()
print('=--=', descs, '=--=')
'''
        output = run_script(script, capture_output=True)
        descs = output.split(b'=--=')[1]
        descs = descs.decode()
        descs = descs.splitlines()
        descs = [parse_descriptor_line(line) for line in descs]
        descs = {x[0]: Descriptor(*x[1:]) for x in descs if x is not None}
        print(descs)
        return descs
