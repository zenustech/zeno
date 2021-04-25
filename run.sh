#!/bin/bash

export PYTHONPATH=`pwd`/python
python -m zenweb &
python -m zensim
kill %1
wait
