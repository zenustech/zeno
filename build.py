#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import subprocess


if sys.platform == 'win32':
    tcpath = 'C:\\Users\\archibate\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake'
    if not os.path.exists(tcpath):
        tcpath = None
else:
    tcpath = None


ap = argparse.ArgumentParser()
ap.add_argument('--config', default='Release')
ap.add_argument('--toolchain', default=tcpath)
ap.add_argument('--clean', action='store_true')
ap.add_argument('--with-openvdb', action='store_true')
ap.add_argument('--with-cuda', action='store_true')
ap.add_argument('--with-bullet', action='store_true')
ap.add_argument('--with-cgal', action='store_true')
ap.add_argument('--build-tests', action='store_true')
ap.add_argument('--build-launcher', action='store_true')
ap.add_argument('--cmake-args', default='')
ap.add_argument('--parallel', default='auto')

ap = ap.parse_args()


if ap.clean:
    shutil.rmtree('zenqt/bin', ignore_errors=True)
    shutil.rmtree('build', ignore_errors=True)


args = []
build_args = []

if sys.platform == 'win32':
    build_args.extend(['--config', ap.config])

if ap.parallel:
    if ap.parallel == 'max':
        build_args.extend(['--parallel'])
    if ap.parallel == 'auto':
        from multiprocessing import cpu_count
        build_args.extend(['--parallel', str(cpu_count())])
    else:
        build_args.extend(['--parallel', ap.parallel])

args.append('-DPYTHON_EXECUTABLE=' + sys.executable)

if ap.build_tests:
    args.extend([
    '-DZENO_BUILD_TESTS:BOOL=ON',
    ])

if ap.build_launcher:
    args.extend([
    '-DZENO_BUILD_LAUNCHER:BOOL=ON',
    ])

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

if ap.with_cgal:
    args.extend([
    '-DEXTENSION_cgmesh:BOOL=ON',
    ])

if ap.with_bullet:
    args.extend([
    '-DEXTENSION_Rigid:BOOL=ON',
    '-DEXTENSION_BulletTools:BOOL=ON',
    ])

if ap.config:
    args.append('-DCMAKE_BUILD_TYPE=' + ap.config)
if ap.toolchain:
    args.append('-DCMAKE_TOOLCHAIN_FILE=' + ap.toolchain)

if ap.cmake_args:
    args.extend(ap.cmake_args.split(','))

print('*** config arguments:', args)
subprocess.check_call(['cmake', '-B', 'build'] + args)
print('*** build arguments:', build_args)
subprocess.check_call(['cmake', '--build', 'build'] + build_args)
