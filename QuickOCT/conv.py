#!/usr/bin/env python

import sys

skip = int(sys.argv[1])

with open('galaxy/data/dubinski.tab', 'r') as fin:
    lines = fin.readlines()
    lines = lines[::skip]
    lines = lines[:len(lines) // 4 * 4]

with open('dubinski.obj', 'w') as fout:
    print('#count', len(lines), file=fout)
    for line in lines:
        m, x, y, z, u, v, w = map(float, line.split())
        print('#v_mass', m, file=fout)
        print('v', x, y, z, file=fout)
        print('#v_vel', u, v, w, file=fout)
