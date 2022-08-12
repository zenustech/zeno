#!/usr/bin/env python
import sys
import subprocess

def process(path):
    with open(path, 'r', encoding='utf-8') as f:
        dat = f.read()
    if dat.startswith('\ufeff'):
        dat = dat[1:]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dat)

if len(sys.argv) > 1:
    process(sys.argv[1])
else:
    out = subprocess.check_output(['bash', '-c', r"find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?'"])
    for x in out.decode().splitlines():
        process(x)
