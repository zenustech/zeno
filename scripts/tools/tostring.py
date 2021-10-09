#!/usr/bin/env python

import sys


with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        line = repr(line + "'")
        line = line[:-2] + '"'
        print(line)
