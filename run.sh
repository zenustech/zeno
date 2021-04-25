#!/bin/bash

export PYTHONPATH=`pwd`/python
kill `lsof -i tcp:8000 | awk '{print $2}' | grep -v PID | uniq`
python -m zenweb &
python -m zensim
kill %1
wait
