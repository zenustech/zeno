#!/usr/bin/env python3

import os
import sys
import subprocess

assert sys.version_info.major == 3
assert sys.version_info.minor == 9

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

args = [
	'--clean',
	'--with-openvdb',
	'--with-bullet',
	'--with-cgal',
]

subprocess.check_call(['git', 'pull'])
subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
subprocess.check_call([sys.executable, 'build.py'] + args)
subprocess.check_call([sys.executable, 'dist.py'])
print('success, thank you zhouhang for help me releasing zeno! now check dist/*.zip')
