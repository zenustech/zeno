#pragma once

#include <memory>
#include <vector>
#include <zeno/core/IObject.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/PolymorphicMap.h>
#include <map>

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
    zeno::PolymorphicMap<std::map<std::string, std::shared_ptr<zeno::IObject>>> objects;
    std::unique_ptr<RenderEngine> renderEngine;

    Scene();
    ~Scene();

    void draw();
    void loadFrameObjects(int frameid);
    void switchRenderEngine(std::string const &name);
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
    bool cameraFocusOnNode(std::string const &nodeid);
    static void loadGLAPI(void *procaddr);
};

} // namespace zenovis
