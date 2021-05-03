#!/bin/bash

# https://stackoverflow.com/questions/7099712/whats-the-accepted-method-for-deploying-a-linux-application-that-relies-on-shar
here=`dirname "$0"`
name=`basename "$0"`

export LD_LIBRARY_PATH="$here"/dsolib:"$LD_LIBRARY_PATH"
export PYTHONPATH="$here"/pythonlib:"$PYTHONPATH"
exec "$here"/dsolib/python "$@"
