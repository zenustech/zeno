#!/usr/bin/env python
# ls zen*/*.{cpp,h} | xargs -n1 python fuckwxl.py
import sys


def process(path):
    with open(path) as f:
        lines = f.readlines()

    with open(path, 'w') as f:
        phase = 0
        for line in lines:
            if phase == 0:
                if line.startswith('<<<<<<<'):
                    phase = 1
                    continue

            if phase == 1:
                if line.startswith('======='):
                    phase = 2
                    continue

            if phase == 2:
                if line.startswith('>>>>>>>'):
                    phase = 0
                    continue

            if phase != 1:
                f.write(line)


process(sys.argv[1])
