#pragma once

#include <cmath>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <variant>
#include <numbers>
#include <optional>
#include <iostream>
#include <functional>
#include <zs/editor/GL/opengl.h>
#include <GLFW/glfw3.h>
#include <FTGL/ftgl.h>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <list>
#include <set>
#include <map>
#include <any>
#include <zs/zeno/ztd/map.h>
#include <zs/zeno/ztd/vector.h>
#include <zs/zeno/ztd/zany.h>


#ifdef __CLANGD__
#define noclangd(...)
#else
#define noclangd(...) __VA_ARGS__
#endif
