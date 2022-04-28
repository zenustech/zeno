#pragma once

#include <memory>
#include <vector>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/PolymorphicVector.h>

namespace zenovis {

struct Camera;
struct DrawOptions;
struct ShaderManager;
struct GraphicsManager;
struct IGraphic;
struct IGraphicDraw;
struct RenderEngine;

struct Scene : zeno::disable_copy {
    std::unique_ptr<Camera> camera;
    std::unique_ptr<DrawOptions> drawOptions;
    std::unique_ptr<ShaderManager> shaderMan;
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::vector<std::unique_ptr<IGraphicDraw>> hudGraphics;
    std::unique_ptr<RenderEngine> renderEngine;

    Scene();
    ~Scene();

    void draw();
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
};

} // namespace zenovis
