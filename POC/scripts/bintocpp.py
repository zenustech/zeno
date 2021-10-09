#!/usr/bin/env python3

import os
import sys

assert len(sys.argv) > 1, 'no file path specified'
with open(sys.argv[1], 'rb') as f:
    print('const unsigned char file_data[] = {')
    for s in f:
        for i in s:
            print(i, end=',')
        print()
    print('};')
