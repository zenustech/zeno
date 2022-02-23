import os
import sys
import subprocess
import shutil

binpath = os.path.join('build', 'bin')

if sys.platform == 'win32':
    for target in ['zenoedit', 'zenorunner']:
        subprocess.check_call([
            '..\\Qt\\5.15.2\\msvc2019_64\\bin\\windeployqt.exe',
            os.path.join(binpath, target + '.exe'),
        ])
    shutil.copyfile(os.path.join('misc', 'ci', 'launch', '000_start.bat'), binpath)
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
    os.mkdir(os.path.join(binpath, 'usr'))
    os.mkdir(os.path.join(binpath, 'usr', 'lib'))
    os.mkdir(os.path.join(binpath, 'usr', 'bin'))
    shutil.copyfile(os.path.join('misc', 'ci', 'launch', '000_start.sh'), binpath)
    shutil.copytree(os.path.join('misc', 'ci', 'share'), os.path.join(binpath, 'usr', 'share'))
    for target in ['zenoedit', 'zenorunner']:
        shutil.move(os.path.join(binpath, target), os.path.join(binpath, 'usr', 'bin', target))
    for target in os.listdir(binpath):
        if 'so' in target.split('.'):
            shutil.move(os.path.join(binpath, target), os.path.join(binpath, 'usr', 'lib', target))
        subprocess.check_call([
            '../linuxdeployqt',
            os.path.join(binpath, 'usr', 'share', ', 'applications', 'zeno.desktop'),
            '-executable=' + os.path.join(binpath, 'usr', 'bin', 'zenorunner')
            '-bundle-non-qt-libs',
        ])
else:
    assert False, sys.platform

shutil.make_archive(binpath, 'zip', binpath, verbose=1)
