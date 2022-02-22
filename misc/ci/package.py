import os
import sys
import subprocess
import shutil

binpath = os.path.join('build', 'bin')
targets = ['zenoedit']

if sys.platform == 'win32':
    for target in targets:
        subprocess.check_call([
            '..\\Qt\\5.15.2\\msvc2019_64\\bin\\windeployqt.exe',
            os.path.join(binpath, target + '.exe'),
        ])
elif sys.platform == 'linux':
    subprocess.check_call([
        'wget',
        'https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage',
        '-O',
        '../linuxdeployqt',
    ])
    subprocess.check_call([
        'chmod',
        '+x',
        '../linuxdeployqt',
    ])
    for target in targets:
        subprocess.check_call([
            '../linuxdeployqt',
            os.path.join(binpath, target),
        ])
else:
    assert False, sys.platform

shutil.make_archive(binpath, 'zip', binpath, verbose=1)
