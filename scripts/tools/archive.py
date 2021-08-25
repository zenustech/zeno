#!/usr/bin/env python

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('file')
ap.add_argument('-n', '--name')
ap.add_argument('-o', '--output')
ap = ap.parse_args()

if ap.name is None:
    ap.name = ap.file
if ap.output is None:
    ap.output = ap.file + '.cpp'

with open(ap.file, 'rb') as f:
    data = f.read()

data = ','.join([str(c) for c in data])

with open(ap.output, 'w') as f:
    print('#include <Hg/Archive.hpp>', file=f)
    print('namespace {', file=f)
    print('static char data[] = {', file=f)
    print(data, file=f)
    print('};', file=f)
    print('static int res = hg::Archive::add('
            '"%s", data, sizeof(data));' % ap.name, file=f)
    print('}', file=f)
