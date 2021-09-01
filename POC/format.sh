#!/bin/bash

find *.cpp demo_project Projects/zenbase Projects/zenvdb Projects/ZMS -regex '.*\.\(c\|h\|hpp\|cpp\)' -exec clang-format --verbose -i *.cpp {} \;
