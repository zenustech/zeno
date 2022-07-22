#!/usr/bin/env python
# find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?' | xargs -n1 python misc/tools/scanutf8.py
import sys
import os


def process(path):
    with open(path, 'r') as f:
        for n, bs in enumerate(f.readlines()):
            for b in map(ord, bs):
                if not 0 <= b <= 0x7f:
                    print('{}:{}: U+{:04x}'.format(path, n + 1, b))
                    exit(1)

# process('zeno/src/nodes/prim/WBPrimBend.cpp')
if len(sys.argv) > 1:
    process(sys.argv[1])
else:
    exit(os.system("find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?' | xargs -n1 python -O misc/tools/scanutf8.py"))
