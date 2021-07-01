#!/bin/bash
set -e

rm -rf /tmp/zenv
mkdir -p /tmp/zenv/{bin,lib/python3.9}

cp -d /tmp/tmp-install/lib/*.so* /tmp/zenv/lib
for x in `scripts/tracedll.sh python3 -m zenapi arts/FLIPSolver.zsg`; do
    y="`realpath $x`"
    echo "$x => $y"
    x="$(echo "$x" | awk -F.so '{print $1".so*"}')"
    cp -d $x /tmp/zenv/lib
    cp "$y" /tmp/zenv/lib
done
cp -d /usr/lib/ld-linux-x86-64.so.2 /usr/lib/ld-2.33.so /tmp/zenv/lib
cp -rd `ls -d /usr/lib/python3.9/* | grep -v site-packages` /tmp/zenv/lib/python3.9
cp -d /usr/bin/python{,3,3.9}{,-config} /tmp/zenv/bin
/tmp/zenv/bin/python3.9 -m ensurepip
https_proxy= python3.9 -m pip install -t /tmp/zenv/lib/python3.9 PyQt5 numpy
/tmp/zenv/bin/python3.9 setup.py install

mv /tmp/zenv/{bin,.bin}
mkdir -p /tmp/zenv/bin

cp scripts/ldmock /tmp/zenv/.ldmock
for x in `ls /tmp/zenv/.bin`; do
    ln -sf ../.ldmock /tmp/zenv/bin/$x
done

cat > /tmp/zenv/start.sh <<EOF
#!/bin/bash

oldwd="\$(pwd)"
cd -- "\$(dirname "\$0")"
newwd="\$(pwd)"
cd -- "\$oldwd"
exec -- "\$newwd/bin/python3.9" -m zenqt "\$@"
EOF
chmod +x /tmp/zenv/start.sh

echo 'docker run -v /tmp/zenv:/tmp/zenv -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -it ubuntu:18.04'
