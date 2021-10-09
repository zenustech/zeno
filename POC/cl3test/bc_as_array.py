#!/usr/bin/env python3

import sys

with open(sys.argv[1], 'rb') as f:
    res = list(f.read())
res = '{' + ','.join(str(x) for x in res) + '},'
print(res)
