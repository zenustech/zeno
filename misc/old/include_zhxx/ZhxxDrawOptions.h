#pragma once

#include <array>
#include <glm/vec3.hpp>

namespace zenovis::zhxx {

struct ZhxxDrawOptions {
    bool passIsDepthPass = true;
    bool passReflect = false;
};

}
