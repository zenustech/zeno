#!/usr/bin/env python
import sys
import locale
import subprocess

failed = False
def process(path):
    try:
        with open(path, 'r') as f:
            for n, bs in enumerate(f.readlines()):
                for b in map(ord, bs):
                    if not 0 <= b <= 0x7f:
                        print('{}:{}: (U+{:04X}) {}'.format(path, n + 1, b, chr(b)))
                        failed = True
    except UnicodeDecodeError:
        print('{}: failed to decode as {}'.format(path, locale.getdefaultlocale()))
        failed = True

if len(sys.argv) > 1:
    process(sys.argv[1])
else:
    out = subprocess.check_output(['bash', '-c', r"find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?'"])
    for x in out.decode().splitlines():
        process(x)
if failed:
    exit(1)
