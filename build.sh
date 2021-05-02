#!/bin/bash

export PYTHONPATH=`pwd`/python
python python/setup.py bdist_wheel
echo - done with python/dist/*.whl
