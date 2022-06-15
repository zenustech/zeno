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
#include "../../xinxinoptix/OptiXStuff.h"
#include <zeno/types/PrimitiveTools.h>
namespace zenovis::optx {

struct GraphicsManager {
    Scene *scene;

        struct DetMaterial {
            std::vector<std::shared_ptr<zeno::Texture2DObject>> tex2Ds;
            std::string common;
            std::string shader;
            std::string mtlidkey;
        };
        struct DetPrimitive {
        };

    struct ZxxGraphic : zeno::disable_copy {
        void computeTrianglesTangent(zeno::PrimitiveObject *prim)
        {
            const auto &tris = prim->tris;
            const auto &pos = prim->attr<zeno::vec3f>("pos");
            auto const &nrm = prim->attr<zeno::vec3f>("nrm");
            auto &tang = prim->tris.add_attr<zeno::vec3f>("tang");
            bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
            //printf("!!has_uv = %d\n", has_uv);
            #pragma omp parallel for
            for (size_t i = 0; i < prim->tris.size(); ++i)
            {
                if(has_uv) {
                    const auto &pos0 = pos[tris[i][0]];
                    const auto &pos1 = pos[tris[i][1]];
                    const auto &pos2 = pos[tris[i][2]];
                    zeno::vec3f uv0;
                    zeno::vec3f uv1;
                    zeno::vec3f uv2;
                    if(has_uv)
                    {
                        uv0 = tris.attr<zeno::vec3f>("uv0")[i];
                        uv1 = tris.attr<zeno::vec3f>("uv1")[i];
                        uv2 = tris.attr<zeno::vec3f>("uv2")[i];
                    }
                    else
                    {
                        uv0 = prim->attr<zeno::vec3f>("uv")[tris[i][0]];
                        uv1 = prim->attr<zeno::vec3f>("uv")[tris[i][1]];
                        uv2 = prim->attr<zeno::vec3f>("uv")[tris[i][2]];
                    }
                    auto edge0 = pos1 - pos0;
                    auto edge1 = pos2 - pos0;
                    auto deltaUV0 = uv1 - uv0;
                    auto deltaUV1 = uv2 - uv0;

                    auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

                    zeno::vec3f tangent;
                    tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
                    tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
                    tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
                    //printf("%f %f %f\n", tangent[0], tangent[1], tangent[3]);
                    auto tanlen = zeno::length(tangent);
                    tangent * (1.f / (tanlen + 1e-8));
                    /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
                        zeno::vec3f n = nrm[tris[i][0]], unused;
                        zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
                    } else {
                        tang[i] = tangent * (1.f / tanlen);
                    }*/
                    tang[i] = tangent;
                } else {
                    tang[i] = zeno::vec3f(0);
                    //zeno::vec3f n = nrm[tris[i][0]], unused;
                    //zeno::pixarONB(n, tang[i], unused);
                }
            }
        }
        std::string key;

        std::variant<DetPrimitive, DetMaterial> det;

        explicit ZxxGraphic(std::string key_, zeno::IObject *obj)
        : key(std::move(key_))
        {
            if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj))
            {
                auto isL = prim_in->userData().getLiterial<int>("isL", 0);
                if(isL == 1){
                    zeno::log_info("processing light key {}", key);
                    auto ivD = prim_in->userData().getLiterial<int>("ivD", 0);

                    auto prim = std::make_shared<zeno::PrimitiveObject>();
                    prim->verts.resize(5);

                    auto p0 = prim_in->verts[prim_in->tris[0][0]];
                    auto p1 = prim_in->verts[prim_in->tris[0][1]];
                    auto p2 = prim_in->verts[prim_in->tris[0][2]];
                    auto e1 = p0-p1;
                    auto e2 = p2-p1;
                    auto g_e1 = glm::vec3(e1[0], e1[1], e1[2]);
                    auto g_e2 = glm::vec3(e2[0], e2[1], e2[2]);
                    glm::vec3 g_nor;

                    g_nor = glm::normalize(glm::cross(g_e1, g_e2));
                    auto nor = zeno::vec3f(g_nor.x, g_nor.y, g_nor.z);
                    zeno::vec3f clr;
                    if(prim_in->verts.has_attr("clr")){
                        clr = prim_in->verts.attr<zeno::vec3f>("clr")[0];
                    }else{
                        clr = zeno::vec3f(30000.0f, 30000.0f, 30000.0f);
                    }
                    prim->verts[0] = p1;
                    prim->verts[1] = e1;
                    prim->verts[2] = e2;
                    prim->verts[3] = nor;
                    prim->verts[4] = clr;

                    xinxinoptix::load_light(key, prim->verts[0].data(), prim->verts[1].data(), prim->verts[2].data(),
                                            prim->verts[3].data(), prim->verts[4].data());

                }else{
                    prim_in->add_attr<zeno::vec3f>("uv");
                    bool primNormalCorrect = prim_in->has_attr("nrm") && length(prim_in->attr<zeno::vec3f>("nrm")[0])>1e-5;
                    bool need_computeNormal = !primNormalCorrect || !(prim_in->has_attr("nrm"));
                    if(prim_in->tris.size() && need_computeNormal)
                    {
                        std::cout<<"computing normal\n";
                        zeno::primCalcNormal(prim_in,1);
                    }
                    computeTrianglesTangent(prim_in);
                    auto prim = std::make_shared<zeno::PrimitiveObject>();

                    prim->verts.resize(prim_in->tris.size()*3);
                    prim->tris.resize(prim_in->tris.size());
                    auto &att_clr = prim->add_attr<zeno::vec3f>("clr");
                    auto &att_nrm = prim->add_attr<zeno::vec3f>("nrm");
                    auto &att_uv  = prim->add_attr<zeno::vec3f>("uv");
                    auto &att_tan = prim->add_attr<zeno::vec3f>("tang");
                    bool has_uv =   prim_in->tris.has_attr("uv0")&&prim_in->tris.has_attr("uv1")&&prim_in->tris.has_attr("uv2");

                    std::cout<<"size verts:"<<prim_in->verts.size()<<std::endl;
                    auto &in_pos   = prim_in->verts;
                    auto &in_clr   = prim_in->add_attr<zeno::vec3f>("clr");
                    auto &in_nrm   = prim_in->add_attr<zeno::vec3f>("nrm");
                    auto &in_uv    = prim_in->attr<zeno::vec3f>("uv");

                    for(size_t tid=0;tid<prim_in->tris.size();tid++)
                    {
                        //std::cout<<tid<<std::endl;
                        size_t vid = tid*3;
                          prim->verts[vid]         = in_pos[prim_in->tris[tid][0]];
                          prim->verts[vid+1]       = in_pos[prim_in->tris[tid][1]];
                          prim->verts[vid+2]       = in_pos[prim_in->tris[tid][2]];
                              att_clr[vid]         = in_clr[prim_in->tris[tid][0]];
                              att_clr[vid+1]       = in_clr[prim_in->tris[tid][1]];
                              att_clr[vid+2]       = in_clr[prim_in->tris[tid][2]];
                              att_nrm[vid]         = in_nrm[prim_in->tris[tid][0]];
                              att_nrm[vid+1]       = in_nrm[prim_in->tris[tid][1]];
                              att_nrm[vid+2]       = in_nrm[prim_in->tris[tid][2]];
                              att_uv[vid]          = has_uv?prim_in->tris.attr<zeno::vec3f>("uv0")[tid]:in_uv[prim_in->tris[tid][0]];
                              att_uv[vid+1]        = has_uv?prim_in->tris.attr<zeno::vec3f>("uv1")[tid]:in_uv[prim_in->tris[tid][1]];
                              att_uv[vid+2]        = has_uv?prim_in->tris.attr<zeno::vec3f>("uv2")[tid]:in_uv[prim_in->tris[tid][2]];
                              att_tan[vid]         = prim_in->tris.attr<zeno::vec3f>("tang")[tid];
                              att_tan[vid+1]       = prim_in->tris.attr<zeno::vec3f>("tang")[tid];
                              att_tan[vid+2]       = prim_in->tris.attr<zeno::vec3f>("tang")[tid];
                             prim->tris[tid]       = zeno::vec3i(vid, vid+1, vid+2);

                    }
                    //flatten here, keep the rest of codes unchanged.

                    det = DetPrimitive{};
                    auto vs = (float const *)prim->verts.data();
                    std::map<std::string, std::pair<float const *, size_t>> vtab;
                    prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                        vtab[key] = {(float const *)arr.data(), sizeof(arr[0]) / sizeof(float)};
                    });
                    auto ts = (int const *)prim->tris.data();
                    auto nvs = prim->verts.size();
                    auto nts = prim->tris.size();
                    auto mtlid = prim_in->userData().getLiterial<std::string>("mtlid", "Default");
                    xinxinoptix::load_object(key, mtlid, vs, nvs, ts, nts, vtab);
                }
            }
            else if (auto mtl = dynamic_cast<zeno::MaterialObject *>(obj))
            {
                det = DetMaterial{mtl->tex2Ds, mtl->common, mtl->frag, mtl->mtlidkey};
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

        bool changelight = false;
        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key)) {
                changelight = true;
            }
        }
        if(changelight){
            xinxinoptix::unload_light();
        }

        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key)) {
                zeno::log_info("zxx_load_object: loading graphics [{}]", key);
                auto ig = std::make_unique<ZxxGraphic>(key, obj);
                zeno::log_info("zxx_load_object: loaded graphics to {}", ig.get());
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
    std::string_view commontpl;
    std::pair<std::string_view, std::string_view> shadtpl2;

    void ensure_shadtmpl() {
        if (ensuredshadtmpl) return;
        ensuredshadtmpl = true;
        shadtmpl = sutil::lookupIncFile("DeflMatShader.cu");
        std::string_view tplsv = shadtmpl;
        std::string_view tmpcommon = "//COMMON_CODE";
        auto pcommon = tplsv.find(tmpcommon);
        auto pcomend = pcommon;
        if(pcommon != std::string::npos)
        {
            pcomend = pcommon + tmpcommon.size();
            commontpl = tplsv.substr(0, pcommon);
        }
        else{
            throw std::runtime_error("cannot find stub COMMON_CODE in shader template");
        }
        std::string_view tmplstub0 = "//GENERATED_BEGIN_MARK";
        std::string_view tmplstub1 = "//GENERATED_END_MARK";
        if (auto p0 = tplsv.find(tmplstub0); p0 != std::string::npos) {
            auto q0 = p0 + tmplstub0.size();
            if (auto p1 = tplsv.find(tmplstub1, q0); p1 != std::string::npos) {
                auto q1 = p1 + tmplstub1.size();
                shadtpl2 = {tplsv.substr(pcomend, p0-pcomend), tplsv.substr(q1)};
            } else {
                throw std::runtime_error("cannot find stub GENERATED_END_MARK in shader template");
            }
        } else {
            throw std::runtime_error("cannot find stub GENERATED_BEGIN_MARK in shader template");
        }
        
    }

    std::map<std::string, int> mtlidlut;

    void draw() override {
        //std::cout<<"in draw()"<<std::endl;
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
            std::vector<std::vector<std::string>> shader_tex_names;
            mtlidlut.clear();

            ensure_shadtmpl();
            shaders.push_back(shadtmpl);
            mtlidlut.insert({"Default", 0});
            shader_tex_names.clear();
            shader_tex_names.push_back(std::vector<std::string>());
            for (auto const &[key, obj]: graphicsMan->graphics) {
                if (auto mtldet = std::get_if<GraphicsManager::DetMaterial>(&obj->det)) {
                    //zeno::log_debug("got material shader:\n{}", mtldet->shader);
                    std::string shader;
                    auto common_code = mtldet->common;
                    std::string tar = "uniform sampler2D";
                    size_t index = 0;
                    while (true) {
                        /* Locate the substring to replace. */
                        index = common_code.find(tar, index);
                        if (index == std::string::npos) break;

                        /* Make the replacement. */
                        common_code.replace(index, tar.length(), "//////////");

                        /* Advance index forward so the next iteration doesn't pick it up as well. */
                        index += tar.length();
                    }

                    shader.reserve(commontpl.size()
                                    + common_code.size()
                                    + shadtpl2.first.size()
                                    + mtldet->shader.size()
                                    + shadtpl2.second.size());
                    shader.append(commontpl);
                    shader.append(common_code);
                    shader.append(shadtpl2.first);
                    shader.append(mtldet->shader);
                    shader.append(shadtpl2.second);
                    //std::cout<<shader<<std::endl;
                    mtlidlut.insert({mtldet->mtlidkey, (int)shaders.size()});
                    shader_tex_names.push_back(std::vector<std::string>());
                    auto &shaderTex = shader_tex_names.back();
                    shaderTex.resize(0);
                    int texid=0;
                    for(auto tex:mtldet->tex2Ds)
                    {
                        OptixUtil::addTexture(tex->path.c_str());
                        shaderTex.emplace_back(tex->path);
                        texid++;
                    }
                    shaders.push_back(std::move(shader));
                    
                }
            }
            std::cout<<"shaders size "<<shaders.size()<<" shader tex name size "<<shader_tex_names.size()<<std::endl;
            xinxinoptix::optixupdatematerial(shaders, shader_tex_names);
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
