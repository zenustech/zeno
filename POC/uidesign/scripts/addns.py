#!/usr/bin/env python3

import os
import sys

assert len(sys.argv) > 1, 'no namespace name specified'
ns = sys.argv[1]
paths = sys.argv[2:]
assert len(paths), 'no path specified'

for path in paths:
    ret = ''
    had_ns = False
    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip('\r\n')
            if not has_ns and line.startswith('#') or not line.strip():
                line = 'namespace ' + ns + ' {'
                had_ns = True

            ret += line + '\n'

    if had_ns:
        line = '}  // namespace ' + ns
        ret += line + '\n'

    with open(path, 'w') as f:
        f.write(ret)
