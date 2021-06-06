#!/usr/bin/env python

import os
import sys
import shutil
import setuptools
import subprocess
from glob import glob


os.chdir(os.path.dirname(os.path.abspath(__file__)))


name = 'zensim'
version = '0.0.1.dev1'
description = 'Open-source node system framework for physics simulation and other CG applications'

packages = ['zenqt', 'zenutils', 'zen', 'zenapi']
requirements = ['numpy', 'PyQt5']

def treefiles(dir):
    if not os.path.isdir(dir):
        yield dir
    else:
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            yield from treefiles(path)

data_files = glob('zen/*.so')
data_files += glob('zen/*.pyd')
data_files += treefiles('zen/usr')
data_files += treefiles('zen/autoload')

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
        #license='Apache 2.0',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            #'Topic :: Software Development :: Compilers',
            'Topic :: Multimedia :: Graphics',
            'Topic :: Games/Entertainment :: Simulation',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            #'License :: OSI Approved :: Apache 2.0 License',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        entry_points={
            'console_scripts': [
                'zenqt=zenqt.main:main',
            ],
        },
        zip_safe=False,
)
