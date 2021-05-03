#!/bin/bash
set -e
cd `dirname "$0"`

rm -rf dist /tmp/dist-zeno
mkdir /tmp/dist-zeno
ln -s /tmp/dist-zeno dist

mkdir dist/lib
cp /usr/bin/python dist/lib
cp `pwd`/build/FastFLIP/libFLIPlib.so dist/lib
python scripts/linkdeps.py dist/lib
cp `realpath /usr/lib/ld-linux-x86-64.so.2` dist/lib/ld-linux.so

mkdir dist/pythonlib
cp -r python/* dist/pythonlib
for x in `ls dist/pythonlib/zenlibs/pydlib/*`; do
    y=`readlink $x`
    rm $x
    cp `pwd`/python/zenlibs/pydlib/$y $x
done

mkdir dist/lib/python3.9
for x in `ls -d /usr/lib/python3.9/* | egrep -v '(site-packages|__pycache__)'`; do
    echo copying $x...
    cp -r $x dist/lib/python3.9
done
cat > dist/lib/python3.9/sitecustomize.py << EOF
import sys
import os

if 'PYTHONEXEC' in os.environ:
    sys.executable = os.environ['PYTHONEXEC']
EOF

cp scripts/python_wrapper.sh dist/python
chmod +x dist/python
