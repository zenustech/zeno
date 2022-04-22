import sys
import subprocess

if sys.platform == 'linux':
    subprocess.check_call([
        'sudo', 'apt-get', 'update', '-y',
    ])
    subprocess.check_call([
        'sudo', 'apt-get', 'install', '-y',
        'autoconf-archive',
    ])

