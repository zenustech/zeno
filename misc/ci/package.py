import os
import sys
import subprocess
import shutil

binpath = os.path.join('build', 'bin')
if sys.platform == 'windows':
    subprocess.check_call([
        'windeployqt',
        os.path.join(binpath, 'zenoedit.exe'),
    ])
elif sys.platform == 'linux':
    subprocess.check_call([
        'wget',
        'https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage',
    ])
    subprocess.check_call([
        './linuxdeployqt-continuous-x86_64.AppImage',
        os.path.join(binpath, 'zenoedit'),
    ])
else:
    assert False, sys.platform

shutil.make_archive(binpath, 'zip', binpath, verbose=1)
