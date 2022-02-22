import os
import sys
import subprocess
import shutil

binpath = os.path.join('build', 'bin')
targets = ['zenoedit']

if sys.platform == 'windows':
    for target in targets:
        subprocess.check_call([
            'windeployqt',
            os.path.join(binpath, target + '.exe'),
        ])
elif sys.platform == 'linux':
    subprocess.check_call([
        'wget', '-c',
        'https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage',
        '-O', '../linuxdeployqt',
    ])
    for target in targets:
        subprocess.check_call([
            '../linuxdeployqt',
            os.path.join(binpath, target),
        ])
else:
    assert False, sys.platform

shutil.make_archive(binpath, 'zip', binpath, verbose=1)
