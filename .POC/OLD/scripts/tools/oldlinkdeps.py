#!/usr/bin/env python

import subprocess as sp
import shutil
import sys
import os


def ldd(file):
    res = []
    p = sp.Popen(['ldd', file], stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    for line in stdout.decode().splitlines():
        try:
            line = line.split(' (')[0]
            line = line.split('\t')[1]
        except ValueError:
            continue
        try:
            name, path = line.split(' => ')
        except ValueError:
            continue
        res.append((name, path))
    return res


os.chdir(sys.argv[1])


had = False
while True:
    for file in os.listdir():
        if not os.path.isfile(file):
            continue
        print('')
        print(file, ':', sep='')
        for name, path in ldd(file):
            assert path != 'not found', (file, name)
            if not os.path.exists(name):
                path = os.path.realpath(path)
                print('  ', name, '- copying', path)
                shutil.copy(path, name)
                had = True
            else:
                print('  ', name, '- exists')
    if not had:
        break
    had = False
