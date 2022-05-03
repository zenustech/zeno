import os
import sys
import subprocess
import shutil

if sys.platform == 'linux':
    print('linux detected')
    subprocess.check_call([
        'sudo', 'apt-get', 'update', '-y',
    ])
    subprocess.check_call([
        'sudo', 'apt-get', 'install', '-y',
        'autoconf-archive', 'libcgal-dev',
    ])
elif sys.platform == 'win32':
    print('windows detected, nothing to do')
else:
    assert False, sys.platform

shutil.move(os.path.join('misc', 'ci', 'vcpkg.json'), 'vcpkg.json')
shutil.rmtree(os.path.join('build', 'CMakeCache.txt'), ignore_errors=True)
shutil.rmtree(os.path.join('build', 'CMakeFiles'), ignore_errors=True)
