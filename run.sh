#!/bin/bash

export PYTHONPATH=`pwd`/python
build/zenvis/zenvis &
python -m zeneditor
wait
