import sys
import json

from . import launchScene

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            scene = json.load(f)
        nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        if 'graph' in scene:
            scene = scene['graph']
        if 'main' not in scene:
            scene = {'main': scene}
        launchScene(scene, nframes)
    else:
        print('Please specify the .zsg file name to run', file=sys.stderr)
    return 0
