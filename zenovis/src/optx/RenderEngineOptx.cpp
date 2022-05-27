#ifdef ZENO_ENABLE_OPTIX
#include "../../xinxinoptix/xinxinoptixapi.h"
#include "../../xinxinoptix/SDK/sutil/sutil.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/UserData.h>
#include <zenovis/DrawOptions.h>
#include <zeno/types/MaterialObject.h>
#include <zenovis/ObjectsManager.h>
#include <zeno/utils/UserData.h>
#include <zeno/utils/fileio.h>
#include <zenovis/RenderEngine.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/scope.h>
#include <zenovis/opengl/vao.h>
#include <variant>

namespace zenovis::optx {

struct GraphicsManager {
    Scene *scene;

        struct DetMaterial {
            std::string shader;
            std::string mtlidkey;
        };
        struct DetPrimitive {
        };

    struct ZxxGraphic : zeno::disable_copy {
        std::string key;

        std::variant<DetPrimitive, DetMaterial> det;

        explicit ZxxGraphic(std::string key_, zeno::IObject *obj)
        : key(std::move(key_))
        {
            if (auto prim = dynamic_cast<zeno::PrimitiveObject *>(obj))
            {
                det = DetPrimitive{};
                auto vs = (float const *)prim->verts.data();
                std::map<std::string, std::pair<float const *, size_t>> vtab;
                prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                    vtab[key] = {(float const *)arr.data(), sizeof(arr[0])};
                });
                auto ts = (int const *)prim->tris.data();
                auto nvs = prim->verts.size();
                auto nts = prim->tris.size();
                auto mtlid = prim->userData().getLiterial<std::string>("mtlid", "Default");
                xinxinoptix::load_object(key, mtlid, vs, nvs, ts, nts, vtab);
            }
            else if (auto mtl = dynamic_cast<zeno::MaterialObject *>(obj))
            {
                det = DetMaterial{mtl->frag, mtl->mtlidkey};
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

    bool lightNeedUpdate = true;
    bool meshNeedUpdate = true;
    bool matNeedUpdate = true;

    auto setupState() {
        return std::tuple{
            opengl::scopeGLEnable(GL_BLEND, false),
            opengl::scopeGLEnable(GL_DEPTH_TEST, false),
            opengl::scopeGLEnable(GL_MULTISAMPLE, false),
        };
    }

    explicit RenderEngineOptx(Scene *scene_) : scene(scene_) {
        zeno::log_info("OptiX Render Engine started...");
        auto guard = setupState();

        graphicsMan = std::make_unique<GraphicsManager>(scene);

        vao = std::make_unique<opengl::VAO>();

        char *argv[] = {nullptr};
        xinxinoptix::optixinit(std::size(argv), argv);
    }

    void update() override {
        if (graphicsMan->load_objects(scene->objectsMan->pairs())) {
            lightNeedUpdate = meshNeedUpdate = matNeedUpdate = true;
        }
    }

#define MY_CAM_ID(cam) cam.m_nx, cam.m_ny, cam.m_lodup, cam.m_lodfront, cam.m_lodcenter, cam.m_fov
#define MY_SIZE_ID(cam) cam.m_nx, cam.m_ny
    std::optional<decltype(std::tuple{MY_CAM_ID(std::declval<Camera>())})> oldcamid;
    std::optional<decltype(std::tuple{MY_SIZE_ID(std::declval<Camera>())})> oldsizeid;

    bool ensuredshadtmpl = false;
    std::string shadtmpl;
    std::pair<std::string_view, std::string_view> shadtpl2;

    void ensure_shadtmpl() {
        if (ensuredshadtmpl) return;
        ensuredshadtmpl = true;
        shadtmpl = sutil::lookupIncFile("DeflMatShader.cu");
        std::string_view tplsv = shadtmpl;
        std::string_view tmplstub0 = "//GENERATED_BEGIN_MARK";
        std::string_view tmplstub1 = "//GENERATED_END_MARK";
        if (auto p0 = tplsv.find(tmplstub0); p0 != std::string::npos) {
            auto q0 = p0 + tmplstub0.size();
            if (auto p1 = tplsv.find(tmplstub1, q0); p1 != std::string::npos) {
                auto q1 = p1 + tmplstub1.size();
                shadtpl2 = {tplsv.substr(0, p0), tplsv.substr(q1)};
            } else {
                throw std::runtime_error("cannot find stub GENERATED_END_MARK in shader template");
            }
        } else {
            throw std::runtime_error("cannot find stub GENERATED_BEGIN_MARK in shader template");
        }
    }

    std::map<std::string, int> mtlidlut;

    void draw() override {
        std::cout<<"aaaaaaa"<<std::endl;
        auto guard = setupState();
        auto const &cam = *scene->camera;
        auto const &opt = *scene->drawOptions;
        //zeno::log_info("test Optx::draw()");

        bool sizeNeedUpdate = false;
        {
            std::tuple newsizeid{MY_SIZE_ID(cam)};
            if (!oldsizeid || *oldsizeid != newsizeid)
                sizeNeedUpdate = true;
            oldsizeid = newsizeid;
        }

        bool camNeedUpdate = false;
        {
            std::tuple newcamid{MY_CAM_ID(cam)};
            if (!oldcamid || *oldcamid != newcamid)
                camNeedUpdate = true;
            oldcamid = newcamid;
        }

        if (sizeNeedUpdate) {
            zeno::log_debug("[zeno-optix] updating resolution");
        xinxinoptix::set_window_size(cam.m_nx, cam.m_ny);
        }

        if (sizeNeedUpdate || camNeedUpdate) {
        zeno::log_debug("[zeno-optix] updating camera");
        //xinxinoptix::set_show_grid(opt.show_grid);
        //xinxinoptix::set_normal_check(opt.normal_check);
        //xinxinoptix::set_enable_gi(opt.enable_gi);
        //xinxinoptix::set_smooth_shading(opt.smooth_shading);
        //xinxinoptix::set_render_wireframe(opt.render_wireframe);
        //xinxinoptix::set_background_color(opt.bgcolor.r, opt.bgcolor.g, opt.bgcolor.b);
        //xinxinoptix::setDOF(cam.m_dof);
        //xinxinoptix::setAperature(cam.m_aperature);
        auto lodright = glm::normalize(glm::cross(cam.m_lodfront, cam.m_lodup));
        //zeno::log_warn("lodup = {}", zeno::other_to_vec<3>(cam.m_lodup));
        //zeno::log_warn("lodfront = {}", zeno::other_to_vec<3>(cam.m_lodfront));
        //zeno::log_warn("lodright = {}", zeno::other_to_vec<3>(lodright));
        xinxinoptix::set_perspective(glm::value_ptr(lodright), glm::value_ptr(cam.m_lodup), glm::value_ptr(cam.m_lodfront), glm::value_ptr(cam.m_lodcenter), cam.getAspect(), cam.m_fov);
        //xinxinoptix::set_projection(glm::value_ptr(cam.m_proj));
        }

        if (meshNeedUpdate || matNeedUpdate || lightNeedUpdate) {
        //zeno::log_debug("[zeno-optix] updating scene");
            
            
            
            //zeno::log_debug("[zeno-optix] updating material");
            std::vector<std::string> shaders;
            mtlidlut.clear();

            ensure_shadtmpl();
            shaders.push_back(shadtmpl);
            mtlidlut.insert({"Default", 0});

            for (auto const &[key, obj]: graphicsMan->graphics) {
                if (auto mtldet = std::get_if<GraphicsManager::DetMaterial>(&obj->det)) {
                    //zeno::log_debug("got material shader:\n{}", mtldet->shader);
                    std::string shader;
                    shader.reserve(shadtpl2.first.size()
                                    + mtldet->shader.size()
                                    + shadtpl2.second.size());
                    shader.append(shadtpl2.first);
                    shader.append(mtldet->shader);
                    shader.append(shadtpl2.second);
                    mtlidlut.insert({mtldet->mtlidkey, (int)shaders.size()});
                    shaders.push_back(std::move(shader));
                }
            }
            xinxinoptix::optixupdatematerial(shaders);
            //zeno::log_debug("[zeno-optix] updating light");
            xinxinoptix::optixupdatelight();
            //zeno::log_debug("[zeno-optix] updating mesh");
            xinxinoptix::optixupdatemesh(mtlidlut);
            

            xinxinoptix::optixupdateend();
            
            
            
            meshNeedUpdate = false;
            matNeedUpdate = false;
            lightNeedUpdate = false;
        }

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
