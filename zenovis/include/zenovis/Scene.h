#pragma once

#include <memory>
#include <vector>
#include <zeno/core/IObject.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/PolymorphicVector.h>

namespace zenovis {

struct Camera;
struct DrawOptions;
struct ShaderManager;
struct GraphicsManager;
struct ObjectsManager;
struct RenderEngine;

struct Scene : zeno::disable_copy {
    std::unique_ptr<Camera> camera;
    std::unique_ptr<DrawOptions> drawOptions;
    std::unique_ptr<ShaderManager> shaderMan;
    std::vector<std::shared_ptr<zeno::IObject>> objects;
    std::unique_ptr<RenderEngine> renderEngine;

    Scene();
    ~Scene();

    void draw();
    void switchRenderEngine(std::string const &name);
    void setObjects(std::vector<std::shared_ptr<zeno::IObject>> const &objs);
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
    void cameraFocusOnNode(std::string const &nodeid);
    static void loadGLAPI(void *procaddr);
};

} // namespace zenovis
