#!/usr/bin/env python3

import os
import sys

paths = sys.argv[1:]
assert len(paths), 'no path specified'

for path in paths:
    ret = ''
    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip('\r\n')
            if line.startswith('#include "'):
                inc = line.split('"')[1]
                inc = os.path.join(os.path.dirname(path), inc)
                line = '#include <' + inc + '>'
            ret += line + '\n'
    with open(path, 'w') as f:
        f.write(ret)
