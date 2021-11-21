#!/bin/bash
set -e
cd `dirname "$0"`

rm -rf dist /tmp/dist-zeno
mkdir /tmp/dist-zeno
ln -s /tmp/dist-zeno dist

mkdir dist/lib
cp /usr/bin/python dist/lib
#for x in libffi.so.7 libGL.so.1 libOpenGL.so.0 libGLEW.so.2.2 libglfw.so.3; do
#    cp `realpath /usr/lib/$x` dist/lib/$x
#done
cp build/FastFLIP/libFLIPlib.so dist/lib
python scripts/linkdeps.py dist/lib
cp `realpath /usr/lib/ld-linux-x86-64.so.2` dist/lib/ld-linux.so

mkdir dist/pythonlib
cp -r python/* dist/pythonlib
for x in `ls dist/pythonlib/zenlibs/pydlib/*`; do
    y=`readlink $x`
    rm $x
    cp python/zenlibs/pydlib/$y $x
done

mkdir dist/lib/python3.9
for x in `ls -d /usr/lib/python3.9/* | egrep -v '(site-packages|__pycache__)'`; do
    echo copying $x...
    cp -r $x dist/lib/python3.9
done
mkdir dist/lib/python3.9/site-packages
cat > dist/lib/python3.9/sitecustomize.py << EOF
import sys
import os

if 'PYTHONEXEC' in os.environ:
    sys.executable = os.environ['PYTHONEXEC']
EOF

cp scripts/python_wrapper.sh dist/python
chmod +x dist/python
dist/python -m ensurepip
dist/python -m pip install -t dist/lib/python3.9/site-packages -U -r python/requirements.txt

x=`pwd`
cd dist
tar zcvf $x/build/zensim.tar.gz .
cd ..
echo DONE WITH build/zensim.tar.gz
