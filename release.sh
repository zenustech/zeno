#!/bin/bash
set -e
cd `dirname "$0"`

rm -rf python/zenlibs/dsolib
mkdir python/zenlibs/dsolib
ln -s /usr/bin/python python/zenlibs/dsolib
ln -s `pwd`/build/FastFLIP/libFLIPlib.so python/zenlibs/dsolib
python python/linkdeps.py python/zenlibs/dsolib

rm -rf python/build python/dist
python python/setup.py build
python python/setup.py bdist_wheel
du -h python/dist/*.whl
