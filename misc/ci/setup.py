import os
import sys
import subprocess
import shutil

if sys.platform == 'linux':
    subprocess.check_call([
        'sudo', 'apt-get', 'update', '-y',
    ])
    subprocess.check_call([
        'sudo', 'apt-get', 'install', '-y',
        'autoconf-archive', 'libcgal-dev',
    ])

shutil.move(os.path.join('misc', 'ci', 'vcpkg.json'), 'vcpkg.json')
