#!/bin/bash

export PYTHONPATH=`pwd`/python
python -m zeneditor &
build/zenvis/zenvis
wait
