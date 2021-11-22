cmake_minimum_required(VERSION 3.12)
project(lua-cmake VERSION 5.4.3 LANGUAGES C DESCRIPTION "Lua build for CMake build system")

set(srcFiles
include/lua/lapi.h
include/lua/lauxlib.h
include/lua/lcode.h
include/lua/lctype.h
include/lua/ldebug.h
include/lua/ldo.h
include/lua/lfunc.h
include/lua/lgc.h
include/lua/ljumptab.h
include/lua/llex.h
include/lua/llimits.h
include/lua/lmem.h
include/lua/lobject.h
include/lua/lopcodes.h
include/lua/lopnames.h
include/lua/lparser.h
include/lua/lprefix.h
include/lua/lstate.h
include/lua/lstring.h
include/lua/ltable.h
include/lua/ltests.h
include/lua/ltm.h
include/lua/luaconf.h
include/lua/lua.h
include/lua/lualib.h
include/lua/lundump.h
include/lua/lvm.h
include/lua/lzio.h
src/lapi.c
src/lauxlib.c
src/lbaselib.c
src/lcode.c
src/lcorolib.c
src/lctype.c
src/ldblib.c
src/ldebug.c
src/ldo.c
src/ldump.c
src/lfunc.c
src/lgc.c
src/linit.c
src/liolib.c
src/llex.c
src/lmathlib.c
src/lmem.c
src/loadlib.c
src/lobject.c
src/lopcodes.c
src/loslib.c
src/lparser.c
src/lstate.c
src/lstring.c
src/lstrlib.c
src/ltable.c
src/ltablib.c
src/ltests.c
src/ltm.c
src/lua.c
src/lundump.c
src/lutf8lib.c
src/lvm.c
src/lzio.c
src/onelua.c
)

add_library(lua ${srcFiles})
target_include_directories(lua PRIVATE include/lua)
target_include_directories(lua PUBLIC include)
