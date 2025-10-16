#pragma once

#include <memory>
#include <vector>
#include <zeno/core/IObject.h>
#include <zeno/core/data.h>
#include <zeno/utils/disable_copy.h>
#include <zeno/utils/vec.h>
#include <map>
#include <optional>
#include <unordered_set>
#include <unordered_map>
#include <zeno/types/ListObject.h>

namespace zenovis {

struct Camera;
struct DrawOptions;
struct ShaderManager;
struct GraphicsManager;
struct RenderManager;

enum class PICK_MODE {
    PICK_NONE,
    PICK_OBJECT,
    PICK_VERTEX,
    PICK_LINE,
    PICK_MESH
};

struct Scene : zeno::disable_copy {
    std::optional<zeno::vec4f> select_box = {};
    std::unordered_set<std::string> selected = {};
    std::unordered_map<std::string, std::unordered_set<int>> selected_elements = {};
    std::unique_ptr<Camera> camera;
    std::unique_ptr<DrawOptions> drawOptions;
    std::unique_ptr<ShaderManager> shaderMan;
    std::unique_ptr<RenderManager> renderMan;

    Scene();
    ~Scene();

    void draw(bool record);
    void reload(const zeno::render_reload_info& info);
    void cleanUpScene();
    void cleanupView();
    void set_show_ptnum(bool bShow);
    bool is_show_ptnum() const;
    void switchRenderEngine(std::string const &name);
    std::vector<char> record_frame_offline(int hdrSize = 1, int rgbComps = 3);
    bool cameraFocusOnNode(std::string const &nodeid, zeno::vec3f &center, float &radius);
    static void loadGLAPI(void *procaddr);
    void* getOptixImg(int &w, int &h);

    void convertListObjs(zeno::IObject* objToBeConvert,           //仅展平对象
        std::vector<std::pair<std::string, zeno::IObject*>>& allListItems);
    void convertListObjs(zeno::IObject* objToBeConvert,           //仅展平对象
        std::map<std::string, zeno::IObject*>& allListItems);
    void set_select_mode(PICK_MODE _select_mode);
    PICK_MODE get_select_mode();
private:
    PICK_MODE select_mode = PICK_MODE::PICK_OBJECT;
    bool m_show_ptnum = false;
};

} // namespace zenovis
