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
    int num_samples = 16;

    bool interactive = false;
    std::string hovered_graphic_id;
    std::map<std::string, std::unique_ptr<IGraphicInteractDraw>> interactGraphics;

    std::set<std::unique_ptr<IGraphicDraw>> interactingGraphics;

    glm::vec3 bgcolor{0.23f, 0.23f, 0.23f};
};

}
