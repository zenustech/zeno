#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/vao.h>
#include "../../zhxxvis/zenvisapi.hpp"

namespace zenovis::zhxx {

struct GraphicsManager {
    Scene *scene;

    struct ZxxGraphic : zeno::disable_copy {
        std::string key;

        explicit ZxxGraphic(std::string key_, zeno::IObject *obj) : key(std::move(key_)) {
            zenvis::zxx_load_object(key, obj);
        }

        ~ZxxGraphic() {
            zenvis::zxx_delete_object(key);
        }
    };

    zeno::MapStablizer<std::map<std::string, std::unique_ptr<ZxxGraphic>>> graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    void load_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();
        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key)) {
                zeno::log_debug("zxx_load_object: loading graphics [{}]", key);
                auto ig = std::make_unique<ZxxGraphic>(key, obj);
                zeno::log_debug("zxx_load_object: loaded graphics to {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
    }
};

struct RenderEngineZhxx : RenderEngine, zeno::disable_copy {
    std::unique_ptr<GraphicsManager> graphicsMan;
    Scene *scene;

    RenderEngineZhxx(Scene *scene_) : scene(scene_) {
        zeno::log_info("Zhxx Render Engine started...");
        CHECK_GL(glEnable(GL_BLEND));
        CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        CHECK_GL(glEnable(GL_DEPTH_TEST));
        CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
        CHECK_GL(glEnable(GL_MULTISAMPLE));
        CHECK_GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        CHECK_GL(glPixelStorei(GL_PACK_ALIGNMENT, 1));

        graphicsMan = std::make_unique<GraphicsManager>(scene);

        zenvis::initialize();
        zenvis::setup_env_map("Default");
    }

    void update() override {
        graphicsMan->load_objects(scene->objectsMan->pairs());
    }

    void draw() override {
        auto const &cam = *scene->camera;
        auto const &opt = *scene->drawOptions;
        auto const &zxx = cam.m_zxx;
        zenvis::set_show_grid(opt.show_grid);
        zenvis::set_normal_check(opt.normal_check);
        zenvis::set_smooth_shading(opt.smooth_shading);
        zenvis::set_render_wireframe(opt.render_wireframe);
        zenvis::set_background_color(opt.bgcolor.r, opt.bgcolor.g, opt.bgcolor.b);
        zenvis::setDOF(cam.m_dof);
        zenvis::setAperature(cam.m_aperature);
        zenvis::set_window_size(cam.m_nx, cam.m_ny);
        zenvis::look_perspective(zxx.cx, zxx.cy, zxx.cz, zxx.theta,
                zxx.phi, zxx.radius, zxx.fov, zxx.ortho_mode);
        int targetFBO = 0;
        CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &targetFBO));
        zenvis::new_frame(targetFBO);
    }

    ~RenderEngineZhxx() override {
        zenvis::finalize();
    }
};

}

namespace zenovis {

std::unique_ptr<RenderEngine> makeRenderEngineZhxx(Scene *scene) {
    return std::make_unique<zhxx::RenderEngineZhxx>(scene);
}

}
