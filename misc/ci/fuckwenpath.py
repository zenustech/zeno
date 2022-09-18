#!/usr/bin/env python

import os
import sys

assert sys.platform == 'win32'

ghp = os.environ['GITHUB_PATH']
ret = []
with open(ghp, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.replace('/', '\\')
        ret.append(line)
with open(ghp, 'w') as f:
    f.write('\n'.join(ret))
