#!/bin/bash

cp `ls ~/Codes/zeno-blender/external/zeno/zenvis/*.{cpp,h,hpp} | grep -v python.cpp | grep -v IGraphic.cpp | grep -v zenvisapi.hpp` "$(dirname "$0")"
sed -i 's/void new_frame() {/void new_frame(int _new_frame_fbo) {/' main.cpp
sed -i 's/  paint_graphics();/  paint_graphics(_new_frame_fbo);/' main.cpp

