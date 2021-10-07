#pragma once

#include <z2/ztd/vec.h>


namespace zeno {

template <size_t N, class T>
using vec = z2::ztd::vec<N, T>;

using z2::ztd::vec2f;
using z2::ztd::vec3f;
using z2::ztd::vec4f;
using z2::ztd::vec2i;
using z2::ztd::vec3i;
using z2::ztd::vec4i;
using z2::ztd::vec2I;
using z2::ztd::vec3I;
using z2::ztd::vec4I;

}
