#pragma once

#include <zenovis/bate/IGraphic.h>

#include <array>
#include <set>
#include <glm/vec3.hpp>

namespace zenovis {

struct DrawOptions {
    float brush_size = 10.0f;
    bool show_grid = true;
    bool render_wireframe = false;

    bool enable_gi = false;
    bool uv_mode = false;
    bool smooth_shading = false;
    bool normal_check = false;
    bool simpleRender = false;
    bool needRefresh = false;
    bool updateMatlOnly = false;
    bool updateLightCameraOnly = false;
    int num_samples = 1;
    int msaa_samples = 0;
    bool denoise = false;
    float viewportPointSizeScale = 1;

    std::shared_ptr<IGraphicHandler> handler;

    glm::vec3 bgcolor{0.18f, 0.20f, 0.22f};
};

}
