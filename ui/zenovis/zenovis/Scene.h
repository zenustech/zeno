#pragma once

#include <memory>
#include <vector>
#include <zeno/utils/disable_copy.h>

namespace zenovis {

struct Camera;
struct ShaderManager;
struct Light;
struct IGraphic;
struct DepthPass;
struct ReflectivePass;

struct Scene : zeno::disable_copy {
    std::unique_ptr<Camera> camera;
    std::vector<std::unique_ptr<Light>> lights;

    std::vector<std::unique_ptr<IGraphic>> graphics;
    std::vector<std::unique_ptr<IGraphic>> hudGraphics;

    std::unique_ptr<ShaderManager> shaderMan;
    std::unique_ptr<DepthPass> mDepthPass;
    std::unique_ptr<ReflectivePass> mReflectivePass;

    Scene();
    ~Scene();

    void drawSceneDepthSafe(float aspRatio, bool reflect, float isDepthPass,
                            bool _show_grid = false);
    void my_paint_graphics(float samples, float isDepthPass);
    void draw_small_axis();

    void draw();
};

} // namespace zenovis
