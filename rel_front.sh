#!/bin/bash

export PYTHONPATH=`pwd`/python
rm -rf python/build python/dist
python python/linkdeps.py
python python/setup.py build
python python/setup.py bdist_wheel
du -h python/dist/*.whl
