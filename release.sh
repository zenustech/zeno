#!/bin/bash
set -e
cd `dirname "$0"`

rm -rf dist
mkdir dist
mkdir dist/dsolib
mkdir dist/pythonlib
cp /usr/bin/python dist/dsolib
cp `pwd`/build/FastFLIP/libFLIPlib.so dist/dsolib
scripts/linkdeps.py dist/dsolib
cp -r python/* dist/pythonlib
for x in `ls dist/pythonlib/zenlibs/pydlib/*`; do
    y=`readlink $x`
    rm $x
    cp `pwd`/python/zenlibs/pydlib/$y $x
done
cp /lib/ld-2.33.so dist/ld-linux.so
cp scripts/start.sh dist/start.sh
