#!/usr/bin/env python3

import os
import sys
import shutil
import setuptools
import subprocess
from glob import glob


os.chdir(os.path.dirname(os.path.abspath(__file__)))


name = 'zensim-zeno'
version = '2021.8.4'
description = 'Open-source node system framework for physics simulation and other CG applications'

packages = ['zenqt']
requirements = ['pybind11', 'numpy', 'PySide2']

def treefiles(dir):
    if not os.path.isdir(dir):
        yield dir
    else:
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            yield from treefiles(path)

data_files = []
data_files += glob('zenqt/*.so')
data_files += glob('zenqt/*.dylib')
data_files += glob('zenqt/*.pyd')
data_files += treefiles('zenqt/assets')
data_files += treefiles('zenqt/lib')

print('version:', version)
print('packages:', packages)
print('data_files:', data_files)
print('requirements:', requirements)


setuptools.setup(
        name=name,
        packages=packages,
        version=version,
        description=description,
        author='archibate',
        author_email='1931127624@qq.com',
        url='https://github.com/zensim-dev/zeno',
        install_requires=requirements,
        data_files=data_files,
        include_package_data=True,
        keywords=['graphics', 'simulation'],
        license='MPL-2.0',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Topic :: Software Development :: Frameworks',
            'Topic :: Multimedia :: Graphics',
            'Topic :: Games/Entertainment :: Simulation',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Mozilla Public License 2.0',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        entry_points={
            'console_scripts': [
                'zeno=zeno.main:main',
                'zenqt=zenqt.main:main',
            ],
        },
        zip_safe=False,
)
