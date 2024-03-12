import os
import sys
from glob import glob 
import pathlib
import subprocess
import shutil
from os.path import dirname, join as pjoin
from setuptools import setup, find_packages

print("zeno pyzpc setup script get zpc build directory:", sys.argv[1])
build_dir = sys.argv[1]

def find_file_dir_recursive(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return root

meta = {}
"""
with open(pjoin('pyzpc', '__version__.py')) as f:
    exec(f.read(), meta)

cwd = pathlib.Path().absolute()
base_dir = dirname(__file__)

lib_prefix = '' if os.name == 'nt' else 'lib'
lib_suffix = 'so'
if sys.platform == 'win32':
    lib_suffix = 'dll'
elif sys.platform == 'darwin':
    lib_suffix = 'dylib'
loc_lib_name = f'{lib_prefix}zpc_py_interop.{lib_suffix}'

print("loc_lib_name: ", loc_lib_name)

build_lib_dir = find_file_dir_recursive(loc_lib_name, build_dir)
shared_lib_paths = glob(pjoin(build_dir, '**/*.so'), recursive=True) + \
    glob(pjoin(build_dir, '**/*.dll'), recursive=True) + \
    glob(pjoin(build_dir, '**/*.dylib'), recursive=True)
for path in shared_lib_paths:
    filename = os.path.basename(path)
    print("iterating:", filename)
    # shutil.copy(
    #     path, 
    #     pjoin(out_lib_dir, filename)
    # )
# # os.removedirs(build_dir)
# os.chdir(str(cwd))
"""