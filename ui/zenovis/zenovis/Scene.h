#pragma once

#include <vector>
#include <memory>

namespace zenovis {

struct Camera;
struct Light;
struct IGraphic;

struct Scene {
    std::unique_ptr<Camera> camera;
    std::vector<std::unique_ptr<Light>> lights;
    std::vector<std::unique_ptr<IGraphic>> graphics;
};

}
