#!/usr/bin/env python3

import subprocess
import sys
import os

subprocess.check_call(['cmake', '-B', 'build', '-DZENO_BUILD_TESTS:BOOL=ON'])
subprocess.check_call([sys.executable, 'build.py'])
if sys.platform == 'win32':
    subprocess.check_call([os.path.join('zenqt', 'bin', 'zeno_tests.exe')])
else:
    subprocess.check_call([os.path.join('zenqt', 'bin', 'zeno_tests')])
