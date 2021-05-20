#!/bin/bash
set -e

docker build . -t zeno
xhost +
<<EOF
Please execute:

cd /root
./get-zeno.sh
cd zeno

after login the container.
EOF
docker run -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -it zeno
