#pragma once

#include <zenovis/bate/IGraphic.h>

#include <array>
#include <set>
#include <glm/vec3.hpp>

namespace zenovis {

struct DrawOptions {
    bool show_grid = true;
    bool render_wireframe = false;

    bool enable_gi = false;
    bool smooth_shading = false;
    bool normal_check = false;
    bool simpleRender = false;
    bool needRefresh = false;
    int num_samples = 1;
    int msaa_samples = 0;

    std::shared_ptr<IGraphicHandler> handler;

    glm::vec3 bgcolor{0.18f, 0.20f, 0.22f};
};

}
