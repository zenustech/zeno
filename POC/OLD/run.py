#!/usr/bin/env python3

import os
import sys
import runpy

assert sys.version_info >= (3, 6), 'Python 3.6+ required, got ' + str(sys.version_info.major) + '.' + str(sys.version_info.minor)

repo_dir = os.path.dirname(os.path.abspath(__file__))
print('ZENO repo directory at:', repo_dir)

sys.path.insert(0, repo_dir)
os.environ['PYTHONPATH'] = repo_dir + os.pathsep \
        + os.environ.get('PYTHONPATH', '')
runpy.run_module('zenqt')
