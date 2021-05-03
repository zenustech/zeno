#!/bin/bash
set -e

cd `dirname "$0"`

rm -rf dist
mkdir dist
mkdir dist/lib
mkdir dist/lib-python
ln -s /usr/bin/python dist/lib/
ln -s `pwd`/build/FastFLIP/libFLIPlib.so dist/lib/
python linkdeps.py dist/lib/
cp scripts/wrapper.sh dist/python
chmod +x dist/python
cp -r python/{zen,zenapi,zenvis,zenlibs,zenutils} dist/lib-python/
for x in dist/lib-python/zenlibs/pydlib/*.so
do y=python/zenlibs/pydlib/`readlink $x`
rm $x && cp $y $x
done
for x in dist/lib/*
do y=`readlink $x`
rm $x && cp $y $x
done
ls dist/*
