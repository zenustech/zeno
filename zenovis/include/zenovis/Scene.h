#pragma once

#include <memory>
#include <vector>
#include <zeno/core/IObject.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/vec.h>
#include <map>
#include <optional>
#include <set>
#include <unordered_set>
#include <unordered_map>

namespace zenovis {

struct Camera;
struct DrawOptions;
struct ShaderManager;
struct GraphicsManager;
struct ObjectsManager;
struct RenderManager;

enum class PICK_MODE {
    PICK_NONE,
    PICK_OBJECT,
    PICK_VERTEX,
    PICK_LINE,
    PICK_FACE,
    PICK_FACE_ATTR,
    PAINT,
};

struct Scene : zeno::disable_copy {
    std::optional<zeno::vec4f> select_box = {};
    std::optional<zeno::vec4f> painter_cursor = {};
    std::unordered_set<std::string> selected = {};
    std::unordered_map<std::string, std::set<int>> selected_elements = {};
    std::unordered_map<std::string, std::pair<std::string, std::set<int>>> selected_int_attr = {};
    std::unordered_map<std::string, std::vector<zeno::vec3f>> painter_data = {};
    std::unique_ptr<Camera> camera;
    std::unique_ptr<DrawOptions> drawOptions;
    std::unique_ptr<ShaderManager> shaderMan;
    std::unique_ptr<ObjectsManager> objectsMan;
    std::unique_ptr<RenderManager> renderMan;

    Scene();
    ~Scene();

    void draw(bool record);
    bool loadFrameObjects(int frameid);
    void cleanUpScene();
    void cleanupView();
    void switchRenderEngine(std::string const &name);
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
    bool cameraFocusOnNode(std::string const &nodeid, zeno::vec3f &center, float &radius);
    static void loadGLAPI(void *procaddr);
    void* getOptixImg(int &w, int &h);
    void set_select_mode(PICK_MODE _select_mode);
    PICK_MODE get_select_mode();
private:
    PICK_MODE select_mode = PICK_MODE::PICK_OBJECT;
};

} // namespace zenovis
