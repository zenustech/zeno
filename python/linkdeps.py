#!/usr/bin/env python

import subprocess as sp
import shutil
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


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

shutil.rmtree('zenlibs/dsolib', ignore_errors=True)
os.mkdir('zenlibs/dsolib')
os.chdir('zenlibs/dsolib')

os.symlink('../../../build/FastFLIP/libFLIPlib.so', 'libFLIPlib.so')


had = False
while True:
    for file in os.listdir():
        print('')
        print(file, ':', sep='')
        for name, path in ldd(file):
            assert path != 'not found', (file, name)
            if not os.path.exists(name):
                path = os.path.realpath(path)
                print('  ', name, '- copying', path)
                os.symlink(path, name)
                had = True
            else:
                print('  ', name, '- exists')
    if not had:
        break
    had = False
