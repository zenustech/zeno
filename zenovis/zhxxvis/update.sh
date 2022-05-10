#!/bin/bash

cp `ls ~/Codes/zeno-blender/external/zeno/zenvis/*.{c,h}{,pp} | grep -v python.cpp | grep -v IGraphic.cpp | grep -v zenvisapi.hpp` "$(dirname "$0")"
