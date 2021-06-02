#!/usr/bin/env python

import os
import setuptools
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))


name = 'zensim'
version = '0.0.1'
description = 'Open-source node system framework for physics simulation and other CG applications'

packages = ['zenqt', 'zenutils', 'zen', 'zenapi', 'zenvis']
data_files = glob.glob('zen/mods/*') + ['zen/libzenpy.so', 'zenvis/libzenvis.so']

requirements = ['numpy', 'pybind11']

print('version:', version)
print('packages:', packages)
print('data_files:', data_files)
print('requirements:', requirements)

setuptools.setup(name=name,
                 packages=packages,
                 version=version,
                 description=description,
                 author='archibate',
                 author_email='1931127624@qq.com',
                 url='https://github.com/zensim-dev/zeno',
                 install_requires=requirements,
                 data_files=data_files,
                 keywords=['graphics', 'simulation'],
                 include_package_data=True,
                 classifiers=[
                     'Topic :: Multimedia :: Graphics',
                     'Topic :: Games/Entertainment :: Simulation',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'Programming Language :: Python :: 3.9',
                 ],
                 has_ext_modules=lambda: True)
