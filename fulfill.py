#!/usr/bin/env python3

import subprocess

args = [
'-DCMAKE_BUILD_TYPE=RelWithDebInfo',
'-DZENOFX_ENABLE_OPENVDB:BOOL=ON',
'-DEXTENSION_oldzenbase:BOOL=ON',
'-DEXTENSION_ZenoFX:BOOL=ON',
'-DEXTENSION_Rigid:BOOL=ON',
'-DEXTENSION_FastFLIP:BOOL=ON',
'-DEXTENSION_zenvdb:BOOL=ON',
]

subprocess.check_call(['cmake', '-B', 'build'] + args)
