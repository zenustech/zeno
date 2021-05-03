#!/bin/bash

here=`dirname "$0"`

echo $here
cd $here
rm -rf dist
mkdir dist
mkdir dist/lib
mkdir dist/bin
mkdir dist/lib-python
ln -s /usr/bin/python dist/lib
python linkdeps.py dist/lib
cp scripts/wrapper.sh dist/python
chmod +x dist/python
cp -r python/{zen,zenapi,zenvis,zenlibs,zenutils} dist/lib-python
ls dist/*
