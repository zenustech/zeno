import os
import sys
import subprocess
import shutil

binpath = os.path.join('build', 'bin')
outpath = os.path.join('build', 'out')

os.mkdir(outpath)
if sys.platform == 'win32':
    shutil.move(os.path.join(binpath, 'zenoedit.exe'), os.path.join(outpath, 'zenoedit.exe'))
    for target in os.listdir(binpath):
        if target.endswith('.dll'):
            shutil.move(os.path.join(binpath, target), os.path.join(outpath, target))
    subprocess.check_call([
        '..\\Qt\\5.15.2\\msvc2019_64\\bin\\windeployqt.exe',
        os.path.join(outpath, 'zenoedit.exe'),
    ])
    shutil.copyfile(os.path.join('misc', 'ci', 'launch', '000_start.bat'), os.path.join(outpath, '000_start.bat'))
    shutil.make_archive(outpath, 'zip', outpath, verbose=1)
    print('finished with', outpath + '.zip')
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
    os.mkdir(os.path.join(outpath, 'usr'))
    os.mkdir(os.path.join(outpath, 'usr', 'lib'))
    os.mkdir(os.path.join(outpath, 'usr', 'bin'))
    shutil.copytree(os.path.join('misc', 'ci', 'share'), os.path.join(outpath, 'usr', 'share'))
    for target in ['zenoedit']:
        shutil.move(os.path.join(binpath, target), os.path.join(outpath, 'usr', 'bin', target))
    for target in os.listdir(binpath):
        if 'so' in target.split('.'):
            shutil.move(os.path.join(binpath, target), os.path.join(outpath, 'usr', 'lib', target))
    subprocess.check_call([
        '../linuxdeployqt',
        os.path.join(outpath, 'usr', 'share', 'applications', 'zeno.desktop'),
        #'-executable=' + os.path.join(outpath, 'usr', 'bin', 'zenorunner'),
        '-bundle-non-qt-libs',
    ])
    shutil.copyfile(os.path.join('misc', 'ci', 'launch', '000_start.sh'), os.path.join(outpath, '000_start.sh'))
    subprocess.check_call([
        'chmod',
        '+x',
        os.path.join(outpath, '000_start.sh'),
    ])
    shutil.make_archive(outpath, 'gztar', outpath, verbose=True)
    print('finished with', outpath + '.tar.gz')
else:
    assert False, sys.platform

