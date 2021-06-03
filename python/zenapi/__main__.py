import sys
import json

from .launch import launchGraph


if len(sys.argv) > 1:

    with open(sys.argv[1], 'r') as f:
        graph = json.load(f)

    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    launchGraph(graph, nframes)

else:
    print('Please specify the .zsg file name to run', file=sys.stderr)
