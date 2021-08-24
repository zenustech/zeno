#!/usr/bin/env python3

import sys
import shutil
import subprocess


config = 'Release'

if sys.platform == 'win32':
    tcfile = 'C:\\Users\\archibate\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake'
else:
    tcfile = None

clean = True
with_openvdb = True
with_cuda = False
with_bullet = True


if clean:
    shutil.rmtree('zenqt/bin', ignore_errors=True)
    shutil.rmtree('build', ignore_errors=True)


args = []

args.append(['-DPYTHON_EXECUTABLE=' + sys.executable])

if with_openvdb:
    args.extend([
    '-DZENOFX_ENABLE_OPENVDB:BOOL=ON',
    '-DEXTENSION_oldzenbase:BOOL=ON',
    '-DEXTENSION_ZenoFX:BOOL=ON',
    '-DEXTENSION_FastFLIP:BOOL=ON',
    '-DEXTENSION_FLIPtools:BOOL=ON',
    '-DEXTENSION_zenvdb:BOOL=ON',
    ])

if with_cuda:
    args.extend([
    '-DZFX_ENABLE_CUDA:BOOL=ON',
    '-DEXTENSION_gmpm:BOOL=ON',
    '-DEXTENSION_mesher:BOOL=ON',
    ])

if with_bullet:
    args.extend([
    '-DEXTENSION_Rigid:BOOL=ON',
    ])

if config:
    args.append('-DCMAKE_BUILD_TYPE=' + config)
if tcfile:
    args.append('-DCMAKE_TOOLCHAIN_FILE=' + tcfile)

subprocess.check_call(['cmake', '-B', 'build'] + args)
subprocess.check_call(['cmake', '--build', 'build', '--parallel'])

subprocess.check_call([sys.executable, 'dist.py'])
