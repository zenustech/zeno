#!/bin/bash

cp `ls ~/Codes/zeno-blender/external/zeno/zenvis/*.cpp | grep -v python.cpp | grep -v IGraphic.cpp` "$(dirname "$0")"
