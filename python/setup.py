#!/usr/bin/env python

import os
import sys
import shutil
import setuptools
import subprocess
from glob import glob

#from setuptools.extension import Extension
#from setuptools.command.build_ext import build_ext
#from setuptools.command.build_py import build_py


os.chdir(os.path.dirname(os.path.abspath(__file__)))


name = 'zensim'
version = '0.0.1'
description = 'Open-source node system framework for physics simulation and other CG applications'

packages = ['zenqt', 'zenutils', 'zen', 'zenapi', 'zenvis']
data_files = glob('zen/*.so') + glob('zenvis/*.so')
data_files += glob('zen/usr')

requirements = ['numpy', 'PyQt5']

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
        #license='Apache 2.0 License',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Topic :: Software Development :: Compilers',
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
        #ext_modules=ext_modules,
        #cmdclass={'build_ext': CMakeBulid},
        zip_safe=False,
)
