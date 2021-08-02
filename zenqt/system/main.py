import sys
import json

from . import run

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--dump-descs':
            descs = run.dumpDescriptors()
            print('==<DESCS>==')
            print(descs)
            print('==<DESCS>==')
            return 0
        with open(sys.argv[1], 'r') as f:
            scene = json.load(f)
        nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        iopath = sys.argv[3] if len(sys.argv) > 3 else '/tmp'
        if 'graph' in scene:
            scene = scene['graph']
        if 'main' not in scene:
            scene = {'main': scene}
        run.runScene(scene, nframes, iopath)
    else:
        print('Please specify the .zsg file name to run', file=sys.stderr)
        return 1
    return 0
