#ifdef ZENO_ENABLE_ZHXXVIS
#include <zenovis/RenderEngine.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/vao.h>
#include <zenovis/opengl/scope.h>
#include "../../zhxxvis/zenvisapi.hpp"

namespace zenovis::zhxx {

struct GraphicsManager {
    Scene *scene;

    struct ZxxGraphic : zeno::disable_copy {
        std::string key;
        std::shared_ptr<zeno::IObject> objholder;

        explicit ZxxGraphic(std::string key_, std::shared_ptr<zeno::IObject> const &obj)
            : key(std::move(key_)), objholder(obj) {
            zenvis::zxx_load_object(key, obj.get());
        }

        ~ZxxGraphic() {
            zenvis::zxx_delete_object(key);
        }
    };

    zeno::MapStablizer<std::map<std::string, std::unique_ptr<ZxxGraphic>>> graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    bool load_objects(std::vector<std::pair<std::string, std::shared_ptr<zeno::IObject>>> const &objs) {
        auto ins = graphics.insertPass();
        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key)) {
                zeno::log_debug("zxx_load_object: loading graphics [{}]", key);
                auto ig = std::make_unique<ZxxGraphic>(key, obj);
                zeno::log_debug("zxx_load_object: loaded graphics to {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
        return ins.has_changed();
    }
};

struct RenderEngineZhxx : RenderEngine, zeno::disable_copy {
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::unique_ptr<opengl::VAO> vao;
    Scene *scene;

    bool giWasEnable = false;
    bool giNeedUpdate = false;

    auto setupState() {
        return std::tuple{
            opengl::scopeGLEnable(GL_BLEND), opengl::scopeGLEnable(GL_DEPTH_TEST),
            opengl::scopeGLEnable(GL_PROGRAM_POINT_SIZE),
            opengl::scopeGLBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
            opengl::scopeGLEnable(GL_MULTISAMPLE),
        };
    }

    explicit RenderEngineZhxx(Scene *scene_) : scene(scene_) {
        zeno::log_info("Zhxx Render Engine started...");
        auto guard = setupState();

        graphicsMan = std::make_unique<GraphicsManager>(scene);

        vao = std::make_unique<opengl::VAO>();
        //auto bindVao = opengl::scopeGLBindVertexArray(vao->vao);

        zenvis::initialize();
        zenvis::setup_env_map("Default");
    }

    void update() override {
        if (graphicsMan->load_objects(scene->objectsMan->pairsShared()))
            giNeedUpdate = true;
    }

    void draw(bool _) override {
        auto guard = setupState();
        auto const &cam = *scene->camera;
        auto const &opt = *scene->drawOptions;

        if (!giWasEnable && opt.enable_gi) {
            giNeedUpdate = true;
        }
        giWasEnable = opt.enable_gi;
        if (giNeedUpdate && opt.enable_gi) {
            zeno::log_debug("scene updated, voxelizing...");
            zenvis::requireVoxelize();
        }
        giNeedUpdate = false;

        zenvis::set_show_grid(opt.show_grid);
        zenvis::set_normal_check(opt.normal_check);
        zenvis::set_enable_gi(opt.enable_gi);
        zenvis::set_smooth_shading(opt.smooth_shading);
        zenvis::set_render_wireframe(opt.render_wireframe);
        zenvis::set_background_color(opt.bgcolor.r, opt.bgcolor.g, opt.bgcolor.b);
        zenvis::setDOF(cam.m_dof);
        zenvis::setAperature(cam.m_aperture);
        zenvis::set_window_size(cam.m_nx, cam.m_ny);
//        zenvis::look_perspective(zxx.cx, zxx.cy, zxx.cz, zxx.theta,
//                zxx.phi, zxx.radius, zxx.fov, zxx.ortho_mode);
        int targetFBO = 0;
        CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &targetFBO));
        CHECK_GL(glClearColor(scene->drawOptions->bgcolor.r, scene->drawOptions->bgcolor.g,
                              scene->drawOptions->bgcolor.b, 0.0f));
        {
            auto bindVao = opengl::scopeGLBindVertexArray(vao->vao);
            zenvis::new_frame(targetFBO);
        }
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, targetFBO));
    }

    ~RenderEngineZhxx() override {
        zenvis::finalize();
    }

    void cleanupAssets() override {

    }

    void cleanupWhenExit() override {

    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineZhxx>("zhxx");

}
#endif
