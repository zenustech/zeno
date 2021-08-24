#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import subprocess


ap = argparse.ArgumentParser()
ap.add_argument('--config', default='Release')
ap.add_argument('--toolchain', type=argparse.FileType('r'), default=os.expanduser('~') + '\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake' if sys.platform == 'win32' else None)
ap.add_argument('--with-openvdb', action='store_true')
ap.add_argument('--with-cuda', action='store_true')
ap.add_argument('--with-bullet', action='store_true')
ap.add_argument('--cmake-args', default='')

ap = ap.parse_args()


if clean:
    shutil.rmtree('zenqt/bin', ignore_errors=True)
    shutil.rmtree('build', ignore_errors=True)


args = []

args.append(['-DPYTHON_EXECUTABLE=' + sys.executable])

if ap.with_openvdb:
    args.extend([
    '-DZENOFX_ENABLE_OPENVDB:BOOL=ON',
    '-DEXTENSION_oldzenbase:BOOL=ON',
    '-DEXTENSION_ZenoFX:BOOL=ON',
    '-DEXTENSION_FastFLIP:BOOL=ON',
    '-DEXTENSION_FLIPtools:BOOL=ON',
    '-DEXTENSION_zenvdb:BOOL=ON',
    ])

if ap.with_cuda:
    args.extend([
    '-DZFX_ENABLE_CUDA:BOOL=ON',
    '-DEXTENSION_gmpm:BOOL=ON',
    '-DEXTENSION_mesher:BOOL=ON',
    ])

if ap.with_bullet:
    args.extend([
    '-DEXTENSION_Rigid:BOOL=ON',
    ])

if ap.config:
    args.append('-DCMAKE_BUILD_TYPE=' + ap.config)
if ap.toolchain:
    args.append('-DCMAKE_TOOLCHAIN_FILE=' + ap.toolchain)

args.extend(ap.cmake.split(';'))

subprocess.check_call(['cmake', '-B', 'build'] + args)

if ap.build:
    subprocess.check_call(['cmake', '--build', 'build', '--parallel'])

if ap.dist:
    subprocess.check_call([sys.executable, 'scripts/final_dist.py'])
