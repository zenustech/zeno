#pragma once

#include <array>
#include <glm/vec3.hpp>

namespace zenovis {

struct DrawOptions {
    bool show_grid = true;
    bool render_wireframe = false;

    bool enable_gi = false;
    bool smooth_shading = false;
    bool normal_check = false;

    glm::vec3 bgcolor{0.23f, 0.23f, 0.23f};
};

}
