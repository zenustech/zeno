#!/usr/bin/env python
import sys
import subprocess

def process(path):
    with open(path, 'r', encoding='gbk') as f:
        dat = f.read()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dat)

if len(sys.argv) > 1:
    process(sys.argv[1])
else:
    out = subprocess.check_output(['bash', '-c', r"find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?'"])
    for x in out.decode().splitlines():
        process(x)
