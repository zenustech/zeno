#ifdef ZENO_ENABLE_OPTIX
#include "../../xinxinoptix/xinxinoptixapi.h"
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/ObjectsManager.h>
#include <zeno/utils/fileio.h>
#include <zenovis/RenderEngine.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/scope.h>
#include <zenovis/opengl/vao.h>

namespace zenovis::optx {

struct GraphicsManager {
    Scene *scene;

    struct ZxxGraphic : zeno::disable_copy {
        std::string key;

        explicit ZxxGraphic(std::string key_, zeno::IObject *obj) : key(std::move(key_)) {
            if (auto prim = dynamic_cast<zeno::PrimitiveObject *>(obj)) {
                auto vs = (float const *)prim->verts.data();
                std::map<std::string, std::pair<float const *, size_t>> vtab;
                prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                    vtab[key] = {(float const *)arr.data(), sizeof(arr[0])};
                });
                auto ts = (int const *)prim->tris.data();
                auto nvs = prim->verts.size();
                auto nts = prim->tris.size();
                xinxinoptix::load_object(key, vs, nvs, ts, nts, vtab);
            }
        }

        ~ZxxGraphic() {
            xinxinoptix::unload_object(key);
        }
    };

    zeno::MapStablizer<std::map<std::string, std::unique_ptr<ZxxGraphic>>> graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    bool load_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
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

struct RenderEngineOptx : RenderEngine, zeno::disable_copy {
    std::unique_ptr<GraphicsManager> graphicsMan;
    std::unique_ptr<opengl::VAO> vao;
    Scene *scene;

    bool giWasEnable = false;
    bool giNeedUpdate = false;

    auto setupState() {
        return std::tuple{
            opengl::scopeGLEnable(GL_BLEND, false),
            opengl::scopeGLEnable(GL_DEPTH_TEST, false),
            opengl::scopeGLEnable(GL_MULTISAMPLE, false),
        };
    }

    explicit RenderEngineOptx(Scene *scene_) : scene(scene_) {
        zeno::log_info("Optx Render Engine started...");
        auto guard = setupState();

        graphicsMan = std::make_unique<GraphicsManager>(scene);

        vao = std::make_unique<opengl::VAO>();

        char *argv[] = {nullptr};
        xinxinoptix::optixinit(std::size(argv), argv);
    }

    void update() override {
        if (graphicsMan->load_objects(scene->objectsMan->pairs()))
            giNeedUpdate = true;
    }

    void draw() override {
        auto guard = setupState();
        auto const &cam = *scene->camera;
        auto const &opt = *scene->drawOptions;

        //xinxinoptix::set_show_grid(opt.show_grid);
        //xinxinoptix::set_normal_check(opt.normal_check);
        //xinxinoptix::set_enable_gi(opt.enable_gi);
        //xinxinoptix::set_smooth_shading(opt.smooth_shading);
        //xinxinoptix::set_render_wireframe(opt.render_wireframe);
        //xinxinoptix::set_background_color(opt.bgcolor.r, opt.bgcolor.g, opt.bgcolor.b);
        //xinxinoptix::setDOF(cam.m_dof);
        //xinxinoptix::setAperature(cam.m_aperature);
        xinxinoptix::set_window_size(cam.m_nx, cam.m_ny);
        auto lodright = glm::normalize(glm::cross(cam.m_lodup, cam.m_lodfront));
        xinxinoptix::set_perspective(glm::value_ptr(lodright), glm::value_ptr(cam.m_lodup), glm::value_ptr(cam.m_lodfront), glm::value_ptr(cam.m_lodcenter), cam.getAspect(), cam.m_fov);
        //xinxinoptix::set_projection(glm::value_ptr(cam.m_proj));

        xinxinoptix::optixupdatemesh();
        std::vector<const char *> shaders;
        auto s = zeno::file_get_content("/home/bate/zeno/zenovis/xinxinoptix/optixPathTracer.cu");
        shaders.push_back(s.c_str());
        shaders.push_back(s.c_str());
        shaders.push_back(s.c_str());
        shaders.push_back(s.c_str());
        shaders.push_back(s.c_str());
        xinxinoptix::optixupdatematerial(shaders);
        xinxinoptix::optixupdateend();

        int targetFBO = 0;
        CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &targetFBO));
        {
            auto bindVao = opengl::scopeGLBindVertexArray(vao->vao);
            xinxinoptix::optixrender(targetFBO);
        }
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, targetFBO));
    }

    ~RenderEngineOptx() override {
        xinxinoptix::optixcleanup();
    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineOptx>("optx");

} // namespace zenovis::optx
#endif
