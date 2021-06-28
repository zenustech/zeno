#!/bin/bash
set -e

docker build . -t zeno
xhost +
docker run -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -it zeno
