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
        'libdbus-1-dev', 'libxi-dev', 'libxtst-dev',
    ])
    subprocess.check_call([
        'sudo', 'apt-get', 'install', '-y',
        'autoconf-archive', 'libcgal-dev', 'libxext-dev',
    ])
    subprocess.check_call([
        'sudo', 'apt-get', 'install', '-y',
        'zlib1g-dev', 'libncurses5-dev', 'libgdbm-dev',
        'libnss3-dev', 'libssl-dev', 'libreadline-dev', 
        'libffi-dev', 'libsqlite3-dev', 'libbz2-dev',
    ])
elif sys.platform == 'win32':
    print('windows detected, nothing to do')
else:
    assert False, sys.platform

shutil.move(os.path.join('misc', 'ci', 'vcpkg.json'), 'vcpkg.json')
shutil.move(os.path.join('misc', 'ci', 'CMakePresets.json'), 'CMakePresets.json')

if os.environ.get('CUDA_PATH'):
    cuda_path = os.environ['CUDA_PATH']
    print('cuda path:', cuda_path)
