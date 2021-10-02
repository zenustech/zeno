#!/usr/bin/env python3

import os
import sys

assert len(sys.argv) > 1, 'no namespace name specified'
ns = sys.argv[1]
assert '/' not in ns, ns + ': looks like a path?'
paths = sys.argv[2:]
assert len(paths), 'no path specified'

for path in paths:
    ret = ''
    had_ns = False
    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip('\r\n')
            if not had_ns:
                if not line.startswith('#') and line.strip():
                    ret += 'namespace ' + ns + ' {\n\n\n'
                    had_ns = True

            ret += line + '\n'

    if had_ns:
        ret += '\n\n}  // namespace ' + ns + '\n'

    with open(path, 'w') as f:
        f.write(ret)
