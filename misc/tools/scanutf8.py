#!/usr/bin/env python
import sys
import locale
import os


def process(path):
    try:
        with open(path, 'r') as f:
            for n, bs in enumerate(f.readlines()):
                for b in map(ord, bs):
                    if not 0 <= b <= 0x7f:
                        print('{}:{}: (U+{:04X}) {}'.format(path, n + 1, b, chr(b)))
                        exit(1)
    except UnicodeDecodeError:
        print('{}: failed to decode as {}'.format(path, locale.getdefaultlocale()))
        exit(2)

if len(sys.argv) > 1:
    process(sys.argv[1])
else:
    exit(os.system(r"find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?' | xargs -n1 python -O misc/tools/scanutf8.py"))
