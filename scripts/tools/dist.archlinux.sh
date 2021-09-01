#!/bin/bash
set -e

cd "$(dirname "$(realpath $0)")/.."

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/tmp-install -DEXTENSION_zenvdb:BOOL=ON -DEXTENSION_FastFLIP:BOOL=ON
make -C build -j 32
make -C build install

rm -rf /tmp/build
mkdir -p /tmp/build/{bin,lib/python3.9}

cp -d /tmp/tmp-install/lib/*.so* /tmp/build/lib
for x in `ZEN_SPROC=1 ZEN_DOEXEC=2 ZEN_OPEN=graphs/VDBMonkeyErode.zsg scripts/tracedll.sh python3 -m zenqt`; do
    y="`realpath $x`"
    echo "$x => $y"
    x="$(echo "$x" | sed 's/\.so/@dot@so/g' | awk -F @dot@so '{print $1".so*"}')"
    cp -d $x /tmp/build/lib
    cp "$y" /tmp/build/lib
done
cp -d /usr/lib/ld-linux-x86-64.so.2 /usr/lib/ld-2.33.so /tmp/build/lib
cp -rd `ls -d /usr/lib/python3.9/* | grep -v site-packages` /tmp/build/lib/python3.9
cp -d /usr/bin/python{,3,3.9}{,-config} /tmp/build/bin
/tmp/build/bin/python3.9 -m ensurepip
https_proxy= python3.9 -m pip install -t /tmp/build/lib/python3.9 pybind11 numpy PySide2
/tmp/build/bin/python3.9 setup.py install

mv /tmp/build/{bin,.bin}
mkdir -p /tmp/build/bin

cp scripts/ldmock /tmp/build/.ldmock
for x in `ls /tmp/build/.bin`; do
    ln -sf ../.ldmock /tmp/build/bin/$x
done

cat > /tmp/build/start.sh <<EOF
#!/bin/bash

oldwd="\$(pwd)"
cd -- "\$(dirname "\$0")"
newwd="\$(pwd)"
cd -- "\$oldwd"
exec -- "\$newwd/bin/python3.9" -m zenqt "\$@"
EOF
chmod +x /tmp/build/start.sh


version=`cat setup.py | grep "version =" | awk -F "version = '" '{print $2}' | cut -d\' -f1`
fname=zeno-linux-$version
echo $fname

cp scripts/release_note.md /tmp/build/README.md
cp -r assets /tmp/build/
cp -r graphs /tmp/build/

cd /tmp
rm -rf $fname $fname.tar.gz
mv build $fname
tar zcvf $fname.tar.gz $fname/

mkdir -p /tmp/release
mv $fname.tar.gz /tmp/release/

