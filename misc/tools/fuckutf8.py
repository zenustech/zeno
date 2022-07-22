#!/usr/bin/env python
import sys
import os


def process(path):
    out = ''
    changed = False
    with open(path, 'rb') as f:
        for bs in f:
            for b in bs:
                if not 0 <= b <= 0x7f:
                    # print('{}:{}: 0x{:02X}'.format(path, n + 1, b))
                    out += '\\x{:02X}'.format(b)
                    changed = True
                else:
                    out += chr(b)
    if changed:
        print(path)
        if not os.environ.get('DRYRUN'):
            with open(path, 'w') as f:
                f.write(out)

if len(sys.argv) > 1:
    process(sys.argv[1])
else:
    exit(os.system(r"find projects ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?' | xargs -n1 python -O misc/tools/fuckutf8.py"))
