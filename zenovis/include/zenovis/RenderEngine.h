#pragma once

#include <zenovis/Camera.h>
#include <zenovis/Scene.h>

namespace zenovis {

struct RenderEngine {
    virtual void draw() = 0;

    virtual ~RenderEngine() = default;
};

std::unique_ptr<RenderEngine> makeRenderEngineDefault(Scene *scene);
std::unique_ptr<RenderEngine> makeRenderEngineZhxx(Scene *scene);

}
