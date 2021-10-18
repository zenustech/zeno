#!/usr/bin/env python

import subprocess
import sys


def ldd(*filenames):
    libs = []
    for x in filenames:
        p = subprocess.Popen(['ldd', x], stdout=subprocess.PIPE)
        for x in p.stdout.readlines():
            x = x.decode()
            s = x.split()
            if '=>' in x:
                if len(s) != 3:
                    libs.append(s[2])
            else:
                if len(s) == 2:
                    libs.append(s[0])
    return libs


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('usage: {} filename1 filename2 ...'.format(sys.argv[0]))
    else:
        print('\n'.join(ldd(*sys.argv[1:])))
