#!/bin/bash

cd "$(dirname "$0")"
cp `ls ~/Codes/zeno-blender/external/zeno/zenvis/*.{cpp,h,hpp} | grep -v python.cpp | grep -v IGraphic.cpp | grep -v zenvisapi.hpp` .
sed -i 's/void new_frame() {/void new_frame(int _new_frame_fbo) {/' main.cpp
sed -i 's/  paint_graphics();/  paint_graphics(_new_frame_fbo);/' main.cpp
#sed -i 's/\<GL_FRAMEBUFFER\>/GL_DRAW_FRAMEBUFFER/g' Light.hpp
