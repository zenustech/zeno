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
else:
    subprocess.check_call([
        'linuxdeployqt',
        os.path.join(binpath, 'zenoedit'),
    ])

shutil.make_archive(binpath, 'zip', binpath, verbose=1)
