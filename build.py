#!/usr/bin/env python3

import subprocess
import shutil

shutil.rmtree('zenoblend/bin', ignore_errors=True)
subprocess.check_call(['cmake', '-B', 'build'])
subprocess.check_call(['cmake', '--build', 'build', '--parallel'])
