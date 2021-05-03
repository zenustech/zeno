#!/bin/bash

# https://stackoverflow.com/questions/7099712/whats-the-accepted-method-for-deploying-a-linux-application-that-relies-on-shar
here=$(realpath `dirname "$0"`)
name=`basename "$0"`

export LD_LIBRARY_PATH="$here"/lib:"$LD_LIBRARY_PATH"
export PYTHONPATH="$here"/pythonlib:"$PYTHONPATH"
export PYTHONHOME="$here"
export PYTHONEXEC="$here"/"$name"
exec "$here"/lib/ld-linux.so "$here"/lib/python "$@"
