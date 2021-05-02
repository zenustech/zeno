#!/usr/bin/env python

import os
import setuptools
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

name = 'zensim'
version = '0.0.1'
description = 'The unified node system for CG simulation'

packages = ['zenlibs', 'zen', 'zenapi', 'zenclient']
data_files = glob.glob('zenlibs/*.so')

with open('requirements.txt') as f:
    requirements = f.readlines()

setuptools.setup(name=name,
                 packages=packages,
                 version=version,
                 description=description,
                 author='archibate',
                 author_email='1931127624@qq.com',
                 url='https://github.com/archibate/zeno',
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
