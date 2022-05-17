#pragma once

#include <memory>
#include <vector>
#include <zeno/core/IObject.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/vec.h>
#include <map>
#include <unordered_set>

namespace zenovis {

struct Camera;
struct DrawOptions;
struct ShaderManager;
struct GraphicsManager;
struct ObjectsManager;
struct RenderManager;

struct Scene : zeno::disable_copy {
    std::unordered_set<std::string> selected = {};
    std::unique_ptr<Camera> camera;
    std::unique_ptr<DrawOptions> drawOptions;
    std::unique_ptr<ShaderManager> shaderMan;
    std::unique_ptr<ObjectsManager> objectsMan;
    std::unique_ptr<RenderManager> renderMan;

    Scene();
    ~Scene();

    void draw();
    void loadFrameObjects(int frameid);
    void switchRenderEngine(std::string const &name);
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
    bool cameraFocusOnNode(std::string const &nodeid, zeno::vec3f &center, float &radius);
    static void loadGLAPI(void *procaddr);
};

} // namespace zenovis
