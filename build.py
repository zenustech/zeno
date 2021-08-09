#!/usr/bin/env python3

import subprocess

subprocess.check_call(['cmake', '-B', 'build'])
subprocess.check_call(['cmake', '--build', 'build', '--parallel'])
