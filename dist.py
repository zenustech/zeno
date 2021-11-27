#!/usr/bin/env python3

import os
import sys
import time
import shutil
import subprocess

if sys.platform == 'win32':
    os_name = 'windows'
elif sys.platform == 'linux':
    os_name = 'linux'
else:
    raise AssertionError('not supported platform: {}'.format(sys.platform))

version = int(time.strftime('%Y')), int(time.strftime('%m')), int(time.strftime('%d'))
version = '{}.{}.{}'.format(*version)

print('==> release version={} os_name={}'.format(version, os_name))

#print('==> copying launcher wrapper')
#if os_name == 'linux':
#    shutil.copyfile('zenqt/bin/launcher', 'dist/launcher')
#elif os_name == 'windows':
#    shutil.copyfile('zenqt/bin/launcher.exe', 'dist/launcher.exe')

if os_name == 'linux':
    print('==> copying linux shared libraries')
    subprocess.check_call([sys.executable, 'scripts/linux_dist_helper.py'])
elif os_name == 'windows':
    print('==> removing windows static libraries')
    subprocess.check_call([sys.executable, 'scripts/windows_dist_helper.py'])

print('==> invoking pyinstaller for packaging')
subprocess.check_call([sys.executable, '-m', 'PyInstaller', 'scripts/{}.spec'.format(os_name), '-y'] + sys.argv[1:])

#print('==> moving launcher wrapper')
#if os_name == 'linux':
#    shutil.move('dist/launcher', 'dist/zenqte/launcher')
#    os.system('chmod +x dist/zenqte/launcher')
#elif os_name == 'windows':
#    shutil.move('dist/launcher.exe', 'dist/zenqte/launcher.exe')

print('==> moving zenqte launcher')
if os_name == 'linux':
    shutil.move('dist/zenqte/zenqte', 'dist/zenqte/launcher')
    os.system('chmod +x dist/zenqte/launcher')
elif os_name == 'windows':
    shutil.move('dist/zenqte/zenqte.exe', 'dist/zenqte/launcher.exe')

#print('==> appending version informations')
#with open('dist/zenqte/zenqt/__init__.py', 'a') as f:
#    f.write('\nversion = {}\n'.format(repr(version)))

zipname = 'dist/zeno-{}-{}'.format(os_name, version)
print('==> creating zip archive at {}'.format(zipname))
shutil.make_archive(zipname, 'zip', 'dist/zenqte', verbose=1)
print('==> done with zip archive {}.zip'.format(zipname))
