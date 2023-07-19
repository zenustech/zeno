#ifdef ZENO_ENABLE_OPTIX
#include "../../xinxinoptix/xinxinoptixapi.h"
#include "../../xinxinoptix/SDK/sutil/sutil.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/UserData.h>
#include <zenovis/DrawOptions.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/MatrixObject.h>
#include <zenovis/ObjectsManager.h>
#include <zeno/utils/UserData.h>
#include <zeno/extra/TempNode.h>
#include <zeno/utils/fileio.h>
#include <zenovis/Scene.h>
#include <zenovis/Camera.h>
#include <zenovis/RenderEngine.h>
#include <zenovis/bate/GraphicsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/opengl/scope.h>
#include <zenovis/opengl/vao.h>
#include <zeno/types/UserData.h>
#include "zeno/core/Session.h"
#include <variant>
#include "../../xinxinoptix/OptiXStuff.h"
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/StringObject.h>

#include <map>
#include <string>
#include <string_view>

namespace zenovis::optx {

struct CppTimer {
    void tick() {
        struct timespec t;
        std::timespec_get(&t, TIME_UTC);
        last = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    }
    void tock() {
        struct timespec t;
        std::timespec_get(&t, TIME_UTC);
        cur = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    }
    float elapsed() const noexcept {return cur-last;}
    void tock(std::string_view tag) {
        tock();
        printf("%s: %f ms\n", tag.data(), elapsed());
    }

  private:
    double last, cur;
};

static CppTimer timer, localTimer;

struct GraphicsManager {
    Scene *scene;

        struct DetMaterial {
            std::vector<std::shared_ptr<zeno::Texture2DObject>> tex2Ds;
            std::vector<std::shared_ptr<zeno::TextureObjectVDB>> tex3Ds;
            std::string common;
            std::string shader;
            std::string mtlidkey;
            std::string extensions;
            std::string transform;
        };
        struct DetPrimitive {
            std::shared_ptr<zeno::PrimitiveObject> primSp;
        };

    struct ZxxGraphic : zeno::disable_copy {
        void computeVertexTangent(zeno::PrimitiveObject *prim)
        {
            auto &atang = prim->add_attr<zeno::vec3f>("atang");
            auto &tang = prim->tris.attr<zeno::vec3f>("tang");
            atang.assign(atang.size(), zeno::vec3f(0));
            const auto &pos = prim->attr<zeno::vec3f>("pos");
            for(size_t i=0;i<prim->tris.size();++i)
            {

                auto vidx = prim->tris[i];
                zeno::vec3f v0 = pos[vidx[0]];
                zeno::vec3f v1 = pos[vidx[1]];
                zeno::vec3f v2 = pos[vidx[2]];
                auto e1 = v1-v0, e2=v2-v0;
                float area = zeno::length(zeno::cross(e1, e2)) * 0.5;
                atang[vidx[0]] += area * tang[i];
                atang[vidx[1]] += area * tang[i];
                atang[vidx[2]] += area * tang[i];
            }
#pragma omp parallel for
            for(size_t i=0;i<atang.size();i++)
            {
                atang[i] = atang[i]/(length(atang[i])+1e-6);

            }
        }
        void computeTrianglesTangent(zeno::PrimitiveObject *prim)
        {
            const auto &tris = prim->tris;
            const auto &pos = prim->attr<zeno::vec3f>("pos");
            auto const &nrm = prim->add_attr<zeno::vec3f>("nrm");
            auto &tang = prim->tris.add_attr<zeno::vec3f>("tang");
            bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
            //printf("!!has_uv = %d\n", has_uv);
            if(has_uv) {
                const auto &uv0data = tris.attr<zeno::vec3f>("uv0");
                const auto &uv1data = tris.attr<zeno::vec3f>("uv1");
                const auto &uv2data = tris.attr<zeno::vec3f>("uv2");
#pragma omp parallel for
                for (size_t i = 0; i < prim->tris.size(); ++i) {
                    const auto &pos0 = pos[tris[i][0]];
                    const auto &pos1 = pos[tris[i][1]];
                    const auto &pos2 = pos[tris[i][2]];
                    zeno::vec3f uv0;
                    zeno::vec3f uv1;
                    zeno::vec3f uv2;

                    uv0 = uv0data[i];
                    uv1 = uv1data[i];
                    uv2 = uv2data[i];

                    auto edge0 = pos1 - pos0;
                    auto edge1 = pos2 - pos0;
                    auto deltaUV0 = uv1 - uv0;
                    auto deltaUV1 = uv2 - uv0;

                    auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

                    zeno::vec3f tangent;
                    tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
                    tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
                    tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
                    //printf("tangent:%f %f %f\n", tangent[0], tangent[1], tangent[2]);
                    //zeno::log_info("tangent {} {} {}",tangent[0], tangent[1], tangent[2]);
                    auto tanlen = zeno::length(tangent);
                    tangent *(1.f / (tanlen + 1e-8));
                    /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
                        zeno::vec3f n = nrm[tris[i][0]], unused;
                        zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
                    } else {
                        tang[i] = tangent * (1.f / tanlen);
                    }*/
                    tang[i] = tangent;
                }
            } else {
                const auto &uvarray = prim->attr<zeno::vec3f>("uv");
#pragma omp parallel for
                for (size_t i = 0; i < prim->tris.size(); ++i) {
                    const auto &pos0 = pos[tris[i][0]];
                    const auto &pos1 = pos[tris[i][1]];
                    const auto &pos2 = pos[tris[i][2]];
                    zeno::vec3f uv0;
                    zeno::vec3f uv1;
                    zeno::vec3f uv2;

                    uv0 = uvarray[tris[i][0]];
                    uv1 = uvarray[tris[i][1]];
                    uv2 = uvarray[tris[i][2]];

                    auto edge0 = pos1 - pos0;
                    auto edge1 = pos2 - pos0;
                    auto deltaUV0 = uv1 - uv0;
                    auto deltaUV1 = uv2 - uv0;

                    auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

                    zeno::vec3f tangent;
                    tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
                    tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
                    tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
                    //printf("tangent:%f %f %f\n", tangent[0], tangent[1], tangent[2]);
                    //zeno::log_info("tangent {} {} {}",tangent[0], tangent[1], tangent[2]);
                    auto tanlen = zeno::length(tangent);
                    tangent *(1.f / (tanlen + 1e-8));
                    /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
                        zeno::vec3f n = nrm[tris[i][0]], unused;
                        zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
                        } else {
                        tang[i] = tangent * (1.f / tanlen);
                        }*/
                    tang[i] = tangent;
                }
            }
        }
        std::string key;

        std::variant<DetPrimitive, DetMaterial> det;

        explicit ZxxGraphic(std::string key_, zeno::IObject *obj)
        : key(std::move(key_))
        {
            if (auto const *prim_in0 = dynamic_cast<zeno::PrimitiveObject *>(obj))
            {
                // vvv deepcopy to cihou following inplace ops vvv
                auto prim_in_lslislSp = std::make_shared<zeno::PrimitiveObject>(*prim_in0);
                // ^^^ Don't wuhui, I mean: Literial Synthetic Lazy internal static Local Shared Pointer
                auto prim_in = prim_in_lslislSp.get();

                auto isInst = prim_in->userData().get2<int>("isInst", 0);

                auto is_sphere = prim_in_lslislSp->userData().has("sphere_center");
                if (is_sphere) {

                    auto& ud = prim_in_lslislSp->userData();
                    printf("Before loading sphere %s for ray tracing... \n", key.c_str());
                    
                    auto mtlid = ud.get2<std::string>("mtlid", "Default");
                    auto instID = ud.get2<std::string>("instID", "Default");

                    bool instanced = (instID != "Default" && instID != "");

                    auto sphere_scale = ud.get2<zeno::vec3f>("sphere_scale");
                    auto uniform_scaling = sphere_scale[0] == sphere_scale[1] && sphere_scale[2] == sphere_scale[0];

                    if (uniform_scaling && instanced) { 
                        auto sphere_center = ud.get2<zeno::vec3f>("sphere_center");
                        auto sphere_radius = ud.get2<float>("sphere_radius");
                             sphere_radius *= sphere_scale[0];

                        xinxinoptix::preload_sphere_crowded(key, mtlid, instID, sphere_radius, sphere_center);
                    } else {

                        //zeno::vec4f row0, row1, row2, row3;
                        auto row0 = ud.get2<zeno::vec4f>("sphere_transform_row0");
                        auto row1 = ud.get2<zeno::vec4f>("sphere_transform_row1");
                        auto row2 = ud.get2<zeno::vec4f>("sphere_transform_row2");
                        auto row3 = ud.get2<zeno::vec4f>("sphere_transform_row3");

                        glm::mat4 sphere_transform;
                        auto transform_ptr = glm::value_ptr(sphere_transform);

                        memcpy(transform_ptr, row0.data(), sizeof(float)*4);
                        memcpy(transform_ptr+4, row1.data(), sizeof(float)*4);
                        memcpy(transform_ptr+8, row2.data(), sizeof(float)*4);  
                        memcpy(transform_ptr+12, row3.data(), sizeof(float)*4);

                        xinxinoptix::preload_sphere_transformed(key, mtlid, instID, sphere_transform);
                    }

                    printf("After loading sphere %s for ray tracing... \n", key.c_str());
                    return;
                }

                auto isRealTimeObject = prim_in->userData().get2<int>("isRealTimeObject", 0);
                auto isUniformCarrier = prim_in->userData().has("ShaderUniforms");
                
                if (isInst == 1)
                {
                    if (!prim_in->has_attr("pos"))
                    {
                        prim_in->add_attr<zeno::vec3f>("pos");
                        prim_in->attr<zeno::vec3f>("pos").assign(prim_in->attr<zeno::vec3f>("pos").size(), zeno::vec3f(0, 0, 0));
                    }
                    if (!prim_in->has_attr("nrm"))
                    {
                        prim_in->add_attr<zeno::vec3f>("nrm");
                        prim_in->attr<zeno::vec3f>("nrm").assign(prim_in->attr<zeno::vec3f>("nrm").size(), zeno::vec3f(0, 1, 0));
                    }
                    if (!prim_in->has_attr("uv"))
                    {
                        prim_in->add_attr<zeno::vec3f>("uv");
                        prim_in->attr<zeno::vec3f>("uv").assign(prim_in->attr<zeno::vec3f>("uv").size(), zeno::vec3f(0, 0, 0));
                    }
                    if (!prim_in->has_attr("clr"))
                    {
                        prim_in->add_attr<zeno::vec3f>("clr");
                        prim_in->attr<zeno::vec3f>("clr").assign(prim_in->attr<zeno::vec3f>("clr").size(), zeno::vec3f(1, 1, 1));
                    }
                    if (!prim_in->has_attr("tang"))
                    {
                        prim_in->add_attr<zeno::vec3f>("tang");
                        prim_in->attr<zeno::vec3f>("tang").assign(prim_in->attr<zeno::vec3f>("tang").size(), zeno::vec3f(1, 0, 0));
                    }
                    
                    auto instID = prim_in->userData().get2<std::string>("instID", "Default");
                    auto onbType = prim_in->userData().get2<std::string>("onbType", "XYZ");
                    
                    std::size_t numInsts = prim_in->verts.size();
                    const float *pos = (const float *)prim_in->attr<zeno::vec3f>("pos").data();
                    const float *nrm = (const float *)prim_in->attr<zeno::vec3f>("nrm").data();
                    const float *uv = (const float *)prim_in->attr<zeno::vec3f>("uv").data();
                    const float *clr = (const float *)prim_in->attr<zeno::vec3f>("clr").data();
                    const float *tang = (const float *)prim_in->attr<zeno::vec3f>("tang").data();
                    xinxinoptix::load_inst(key, instID, onbType, numInsts, pos, nrm, uv, clr, tang);
                }
                else if (isRealTimeObject == 0 && isUniformCarrier == 0)
                {
                    //first init matidx attr
                    int matNum = prim_in->userData().get2<int>("matNum",0);
                    if(matNum==0)
                    {
                        //assign -1 to "matid" attr
                        if(prim_in->tris.size()>0) {
                            prim_in->tris.add_attr<int>("matid");
                            prim_in->tris.attr<int>("matid").assign(prim_in->tris.size(), -1);
                        }
                        if(prim_in->quads.size()>0) {
                            prim_in->quads.add_attr<int>("matid");
                            prim_in->quads.attr<int>("matid").assign(prim_in->quads.size(), -1);
                        }
                        if(prim_in->polys.size()>0) {
                            prim_in->polys.add_attr<int>("matid");
                            prim_in->polys.attr<int>("matid").assign(prim_in->polys.size(), -1);
                        }
                    }



        det = DetPrimitive{prim_in_lslislSp};
        if (int subdlevs = prim_in->userData().get2<int>("delayedSubdivLevels", 0)) {
            // todo: zhxx, should comp normal after subd or before????
            zeno::log_trace("computing subdiv {}", subdlevs);
            (void)zeno::TempNodeSimpleCaller("OSDPrimSubdiv")
                .set("prim", prim_in_lslislSp)
                .set2<int>("levels", subdlevs)
                .set2<std::string>("edgeCreaseAttr", "")
                .set2<bool>("triangulate", true)
                .set2<bool>("asQuadFaces", true)
                .set2<bool>("hasLoopUVs", true)
                .set2<bool>("delayTillIpc", false)
                .call();  // will inplace subdiv prim
            prim_in->userData().del("delayedSubdivLevels");
        }
                    if (prim_in->quads.size() || prim_in->polys.size()) {
                        zeno::log_trace("demoting faces");
                        zeno::primTriangulateQuads(prim_in);
                        zeno::primTriangulate(prim_in);
                    }
                    if(prim_in->tris.size()==0) return;

//                    /// WXL
//                    (void)zeno::TempNodeSimpleCaller("PrimitiveReorder")
//                        .set("prim", std::shared_ptr<zeno::PrimitiveObject>(prim_in, [](void *) {}))
//                        .set2<bool>("order_vertices", true)
//                        .set2<bool>("order_tris", true)
//                        .call();  // will inplace reorder prim
//                    /// WXL

                    bool has_uv =   prim_in->tris.has_attr("uv0")&&prim_in->tris.has_attr("uv1")&&prim_in->tris.has_attr("uv2");
                    if(prim_in->has_attr("uv") && has_uv == false)
                    {
                        auto &uv = prim_in->attr<zeno::vec3f>("uv");
                        auto &uv0 = prim_in->tris.add_attr<zeno::vec3f>("uv0");
                        auto &uv1 = prim_in->tris.add_attr<zeno::vec3f>("uv1");
                        auto &uv2 = prim_in->tris.add_attr<zeno::vec3f>("uv2");
                        for(size_t i=0; i<prim_in->tris.size();i++)
                        {
                            uv0[i]=uv[prim_in->tris[i][0]];
                            uv1[i]=uv[prim_in->tris[i][1]];
                            uv2[i]=uv[prim_in->tris[i][2]];
                        }
                    }
                    prim_in->add_attr<zeno::vec3f>("uv");
                    bool primNormalCorrect = prim_in->has_attr("nrm") && length(prim_in->attr<zeno::vec3f>("nrm")[0])>1e-5;
                    bool need_computeNormal = !primNormalCorrect || !(prim_in->has_attr("nrm"));
                    if(prim_in->tris.size() && need_computeNormal)
                    {
                        zeno::log_trace("computing normal");
                        zeno::primCalcNormal(prim_in,1);
                    }
                    computeTrianglesTangent(prim_in);
                    computeVertexTangent(prim_in);
                    auto prim = std::make_shared<zeno::PrimitiveObject>();

                    prim->verts.resize(prim_in->tris.size()*3);
                    prim->tris.resize(prim_in->tris.size());
                    auto &att_clr = prim->add_attr<zeno::vec3f>("clr");
                    auto &att_nrm = prim->add_attr<zeno::vec3f>("nrm");
                    auto &att_uv  = prim->add_attr<zeno::vec3f>("uv");
                    auto &att_tan = prim->add_attr<zeno::vec3f>("tang");
                    has_uv =   prim_in->tris.has_attr("uv0")&&prim_in->tris.has_attr("uv1")&&prim_in->tris.has_attr("uv2");

                    std::cout<<"size verts:"<<prim_in->verts.size()<<std::endl;
                    auto &in_pos   = prim_in->verts;
                    auto &in_tan   = prim_in->attr<zeno::vec3f>("atang");
                    auto &in_nrm   = prim_in->add_attr<zeno::vec3f>("nrm");
                    auto &in_uv    = prim_in->attr<zeno::vec3f>("uv");
                    const zeno::vec3f* uv_data0 = nullptr;
                    const zeno::vec3f* uv_data1 = nullptr;
                    const zeno::vec3f* uv_data2 = nullptr;

                    if(has_uv) {
                        uv_data0 = prim_in->tris.attr<zeno::vec3f>("uv0").data();
                        uv_data1 = prim_in->tris.attr<zeno::vec3f>("uv1").data();
                        uv_data2 = prim_in->tris.attr<zeno::vec3f>("uv2").data();

                        for (size_t tid = 0; tid < prim_in->tris.size(); tid++) {
                            //std::cout<<tid<<std::endl;
                            size_t vid = tid * 3;
                            prim->verts[vid] = in_pos[prim_in->tris[tid][0]];
                            prim->verts[vid + 1] = in_pos[prim_in->tris[tid][1]];
                            prim->verts[vid + 2] = in_pos[prim_in->tris[tid][2]];
                            att_nrm[vid] = in_nrm[prim_in->tris[tid][0]];
                            att_nrm[vid + 1] = in_nrm[prim_in->tris[tid][1]];
                            att_nrm[vid + 2] = in_nrm[prim_in->tris[tid][2]];
                            att_uv[vid] = uv_data0[tid];
                            att_uv[vid + 1] = uv_data1[tid];
                            att_uv[vid + 2] = uv_data2[tid];
                            att_tan[vid] = in_tan[prim_in->tris[tid][0]];
                            att_tan[vid + 1] = in_tan[prim_in->tris[tid][1]];
                            att_tan[vid + 2] = in_tan[prim_in->tris[tid][2]];
                            prim->tris[tid] = zeno::vec3i(vid, vid + 1, vid + 2);
                        }
                    } else
                    {
                        for (size_t tid = 0; tid < prim_in->tris.size(); tid++) {
                            //std::cout<<tid<<std::endl;
                            size_t vid = tid * 3;
                            prim->verts[vid] = in_pos[prim_in->tris[tid][0]];
                            prim->verts[vid + 1] = in_pos[prim_in->tris[tid][1]];
                            prim->verts[vid + 2] = in_pos[prim_in->tris[tid][2]];
                            att_nrm[vid] = in_nrm[prim_in->tris[tid][0]];
                            att_nrm[vid + 1] = in_nrm[prim_in->tris[tid][1]];
                            att_nrm[vid + 2] = in_nrm[prim_in->tris[tid][2]];
                            att_uv[vid] = in_uv[prim_in->tris[tid][0]];
                            att_uv[vid + 1] = in_uv[prim_in->tris[tid][1]];
                            att_uv[vid + 2] = in_uv[prim_in->tris[tid][2]];
                            att_tan[vid] = in_tan[prim_in->tris[tid][0]];
                            att_tan[vid + 1] = in_tan[prim_in->tris[tid][1]];
                            att_tan[vid + 2] = in_tan[prim_in->tris[tid][2]];
                            prim->tris[tid] = zeno::vec3i(vid, vid + 1, vid + 2);
                        }
                    }
                    if (prim_in->has_attr("clr")) {
                        auto &in_clr   = prim_in->add_attr<zeno::vec3f>("clr");
                        for(size_t tid=0;tid<prim_in->tris.size();tid++) {
                            size_t vid = tid*3;
                            att_clr[vid]         = in_clr[prim_in->tris[tid][0]];
                            att_clr[vid+1]       = in_clr[prim_in->tris[tid][1]];
                            att_clr[vid+2]       = in_clr[prim_in->tris[tid][2]];
                        }
                    }
                    //flatten here, keep the rest of codes unchanged.

                    auto vs = (float const *)prim->verts.data();
                    std::map<std::string, std::pair<float const *, size_t>> vtab;
                    prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                        vtab[key] = {(float const *)arr.data(), sizeof(arr[0]) / sizeof(float)};
                    });
                    auto ts = (int const *)prim->tris.data();
                    auto matids = (int const *)prim_in->tris.attr<int>("matid").data();
                    std::vector<std::string> matNameList(0);
                    if(matNum>0)
                    {
                        for(int i=0;i<matNum;i++)
                        {
                            auto matIdx = "Material_" + std::to_string(i);
                            auto matName = prim_in->userData().get2<std::string>(matIdx, "Default");
                            matNameList.emplace_back(matName);
                        }
                    }
                    auto nvs = prim->verts.size();
                    auto nts = prim->tris.size();
                    auto mtlid = prim_in->userData().get2<std::string>("mtlid", "Default");
                    auto instID = prim_in->userData().get2<std::string>("instID", "Default");
                    xinxinoptix::load_object(key, mtlid, instID, vs, nvs, ts, nts, vtab, matids, matNameList);
                }
            }
            else if (auto mtl = dynamic_cast<zeno::MaterialObject *>(obj))
            {
                det = DetMaterial{mtl->tex2Ds, mtl->tex3Ds, mtl->common, mtl->frag, mtl->mtlidkey, mtl->extensions, mtl->transform};
            }
        }

        ~ZxxGraphic() {
            xinxinoptix::unload_object(key);
            xinxinoptix::unload_inst(key);
        }
    };

    zeno::MapStablizer<std::map<std::string, std::unique_ptr<ZxxGraphic>>> graphics;
    std::map<std::string, int> objOrder;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    bool load_shader_uniforms(std::vector<std::pair<std::string, zeno::IObject *>> const &objs)
    {
        std::vector<float4> shaderUniforms;
        shaderUniforms.resize(0);
        for (auto const &[key, obj] : objs) {
            if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj)){
                if ( prim_in->userData().get2<int>("ShaderUniforms", 0)==1 )
                {

                    shaderUniforms.resize(prim_in->verts.size());
                    for(int i=0;i<prim_in->verts.size();i++)
                    {
                        shaderUniforms[i] = make_float4(prim_in->verts[i][0],prim_in->verts[i][1],prim_in->verts[i][2],0);
                    }
                }
            }
        }
        // xinxinoptix::optixUpdateUniforms(shaderUniforms);
        xinxinoptix::optixUpdateUniforms(shaderUniforms.data(), shaderUniforms.size());
        return true;
    }
    // return if find sky
    bool load_lights(std::string key, zeno::IObject *obj){
        bool sky_found = false;
        if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj)) {
            auto isRealTimeObject = prim_in->userData().get2<int>("isRealTimeObject", 0);
            if (isRealTimeObject == 0) {
                return false;
            }
            if (prim_in->userData().get2<int>("isL", 0) == 1) {
                //zeno::log_info("processing light key {}", key.c_str());
                auto ivD = prim_in->userData().getLiterial<int>("ivD", 0);

                auto p0 = prim_in->verts[prim_in->tris[0][0]];
                auto p1 = prim_in->verts[prim_in->tris[0][1]];
                auto p2 = prim_in->verts[prim_in->tris[0][2]];
                auto e1 = p0 - p2;
                auto e2 = p1 - p2;

                auto nor = zeno::normalize(zeno::cross(e1, e2));
                zeno::vec3f clr;
                if (prim_in->verts.has_attr("clr")) {
                    clr = prim_in->verts.attr<zeno::vec3f>("clr")[0];
                } else {
                    clr = zeno::vec3f(30000.0f, 30000.0f, 30000.0f);
                }

                std::cout << "light: p"<<p0[0]<<" "<<p0[1]<<" "<<p0[2]<<"\n";
                std::cout << "light: p"<<p1[0]<<" "<<p1[1]<<" "<<p1[2]<<"\n";
                std::cout << "light: p"<<p2[0]<<" "<<p2[1]<<" "<<p2[2]<<"\n";
                std::cout << "light: e"<<e1[0]<<" "<<e1[1]<<" "<<e1[2]<<"\n";
                std::cout << "light: e"<<e2[0]<<" "<<e2[1]<<" "<<e2[2]<<"\n";
                std::cout << "light: n"<<nor[0]<<" "<<nor[1]<<" "<<nor[2]<<"\n";
                std::cout << "light: c"<<clr[0]<<" "<<clr[1]<<" "<<clr[2]<<"\n";

                xinxinoptix::load_light(key, p2.data(), e1.data(), e2.data(),
                                        nor.data(), clr.data());
            }
            else if (prim_in->userData().get2<int>("ProceduralSky", 0) == 1) {
                sky_found = true;
                zeno::vec2f sunLightDir = prim_in->userData().get2<zeno::vec2f>("sunLightDir");
                float sunLightSoftness = prim_in->userData().get2<float>("sunLightSoftness");
                float sunLightIntensity = prim_in->userData().get2<float>("sunLightIntensity");
                float colorTemperatureMix = prim_in->userData().get2<float>("colorTemperatureMix");
                float colorTemperature = prim_in->userData().get2<float>("colorTemperature");
                zeno::vec2f windDir = prim_in->userData().get2<zeno::vec2f>("windDir");
                float timeStart = prim_in->userData().get2<float>("timeStart");
                float timeSpeed = prim_in->userData().get2<float>("timeSpeed");
                xinxinoptix::update_procedural_sky(sunLightDir, sunLightSoftness, windDir, timeStart, timeSpeed,
                                                   sunLightIntensity, colorTemperatureMix, colorTemperature);
            }
            else if (prim_in->userData().has<std::string>("HDRSky")) {
                auto path = prim_in->userData().get2<std::string>("HDRSky");
                float evnTexRotation = prim_in->userData().get2<float>("evnTexRotation");
                zeno::vec3f evnTex3DRotation = prim_in->userData().get2<zeno::vec3f>("evnTex3DRotation");
                float evnTexStrength = prim_in->userData().get2<float>("evnTexStrength");
                bool enableHdr = prim_in->userData().get2<bool>("enable");
                OptixUtil::sky_tex = path;
                OptixUtil::addTexture(path);
                xinxinoptix::update_hdr_sky(evnTexRotation, evnTex3DRotation, evnTexStrength);
                xinxinoptix::using_hdr_sky(enableHdr);
            }
        }
        return sky_found;
    }

    bool need_update_light(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();

        bool changelight = false;
        for (auto const &[key, obj] : objs) {
            if (scene->drawOptions->updateMatlOnly && ins.may_emplace(key))
            {
                changelight = false;
            }
            else if(ins.may_emplace(key)) {
                changelight = true;
            }
        }

        auto &ud = zeno::getSession().userData();
        bool show_background = ud.get2<bool>("optix_show_background", false);
        xinxinoptix::show_background(show_background);

        return changelight;
    }
    bool load_light_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> objs){
        xinxinoptix::unload_light();
        bool sky_found = false;

        for (auto const &[key, obj] : objs) {
            if(load_lights(key, obj.get())) {
                sky_found = true;
            }
        }
//        zeno::log_info("sky_found : {}", sky_found);
        if (sky_found == false) {
            auto &ud = zeno::getSession().userData();
//            zeno::log_info("ud.has sunLightDir: {}", ud.has("sunLightDir"));
            if (ud.has("sunLightDir")) {
                zeno::vec2f sunLightDir = ud.get2<zeno::vec2f>("sunLightDir");
                float sunLightSoftness = ud.get2<float>("sunLightSoftness");
                float sunLightIntensity = ud.get2<float>("sunLightIntensity");
                float colorTemperatureMix = ud.get2<float>("colorTemperatureMix");
                float colorTemperature = ud.get2<float>("colorTemperature");
                zeno::vec2f windDir = ud.get2<zeno::vec2f>("windDir");
                float timeStart = ud.get2<float>("timeStart");
                float timeSpeed = ud.get2<float>("timeSpeed");
                xinxinoptix::update_procedural_sky(sunLightDir, sunLightSoftness, windDir, timeStart, timeSpeed,
                                                   sunLightIntensity, colorTemperatureMix, colorTemperature);
            }
        }

        return true;
    }

    bool load_static_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();

        bool changed = false;

        for (auto const &[key, obj] : objs) {
            if (ins.may_emplace(key) && key.find(":static:")!=key.npos) {
                zeno::log_info("load_static_object: loading graphics [{}]", key);
                changed = true;

                if (auto cam = dynamic_cast<zeno::CameraObject *>(obj))
                {
                    scene->camera->setCamera(cam->get());     // pyb fix
                }

                auto ig = std::make_unique<ZxxGraphic>(key, obj);

                zeno::log_info("load_static_object: loaded graphics to {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
        // return ins.has_changed();
        return changed;
    }
    bool load_objects(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();
        objOrder.clear();
        bool changed = false;
        size_t idx = 0;
        for (auto const &[key, obj] : objs) {
            objOrder[key] = idx;
            idx++;
        }
        for (auto const &[key, obj] : objs) {
            //auto ikey = key + ':' + std::string(std::to_string(idx));
            if (ins.may_emplace(key) && key.find(":static:")==key.npos) {

                zeno::log_info("load_object: loading graphics [{}]", key);
                changed = true;

                if (!scene->drawOptions->updateMatlOnly) {
                    if (auto cam = dynamic_cast<zeno::CameraObject *>(obj)) {
                        scene->camera->setCamera(cam->get()); // pyb fix
                    }
                }

                auto ig = std::make_unique<ZxxGraphic>(key, obj);

                zeno::log_info("load_object: loaded graphics to {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
        // return ins.has_changed();
        return changed;
    }
};

struct RenderEngineOptx : RenderEngine, zeno::disable_copy {
    std::unique_ptr<GraphicsManager> graphicsMan;
#ifdef OPTIX_BASE_GL
    std::unique_ptr<opengl::VAO> vao;
#endif
    Scene *scene;


    bool lightNeedUpdate = true;
    bool meshNeedUpdate = true;
    bool matNeedUpdate = true;
    bool staticNeedUpdate = true;

    auto setupState() {
        return std::tuple{
            opengl::scopeGLEnable(GL_BLEND, false),
            opengl::scopeGLEnable(GL_DEPTH_TEST, false),
            opengl::scopeGLEnable(GL_MULTISAMPLE, false),
        };
    }

    explicit RenderEngineOptx(Scene *scene_) : scene(scene_) {
        zeno::log_info("OptiX Render Engine started...");
#ifdef OPTIX_BASE_GL
        auto guard = setupState();
#endif

        graphicsMan = std::make_unique<GraphicsManager>(scene);

#ifdef OPTIX_BASE_GL
        vao = std::make_unique<opengl::VAO>();
#endif

        char *argv[] = {nullptr};
        xinxinoptix::optixinit(std::size(argv), argv);
    }

    void update() override {

        if(graphicsMan->need_update_light(scene->objectsMan->pairs())
            || scene->objectsMan->needUpdateLight)
        {
            graphicsMan->load_light_objects(scene->objectsMan->lightObjects);
            lightNeedUpdate = true;
            scene->objectsMan->needUpdateLight = false;
            scene->drawOptions->needRefresh = true;
        }

        if (graphicsMan->load_static_objects(scene->objectsMan->pairs())) {
            staticNeedUpdate = true;
        }
        if (graphicsMan->load_objects(scene->objectsMan->pairs()))
        {
            meshNeedUpdate = matNeedUpdate = true;
            if (scene->drawOptions->updateMatlOnly)
            {
                lightNeedUpdate = meshNeedUpdate = false;
                matNeedUpdate = true;
            }
            if (scene->drawOptions->updateLightCameraOnly)
            {
                lightNeedUpdate = true;
                matNeedUpdate = meshNeedUpdate = false;
            }
        }
        graphicsMan->load_shader_uniforms(scene->objectsMan->pairs());
    }

#define MY_CAM_ID(cam) cam.m_nx, cam.m_ny, cam.m_lodup, cam.m_lodfront, cam.m_lodcenter, cam.m_fov, cam.focalPlaneDistance, cam.m_aperture
#define MY_SIZE_ID(cam) cam.m_nx, cam.m_ny
    std::optional<decltype(std::tuple{MY_CAM_ID(std::declval<Camera>())})> oldcamid;
    std::optional<decltype(std::tuple{MY_SIZE_ID(std::declval<Camera>())})> oldsizeid;

    struct ShaderTemplateInfo {
        const std::string name;

        bool ensured = false;
        std::string shadtmpl {};
        std::string_view commontpl {};
        std::pair<std::string_view, std::string_view> shadtpl2;
    };

    ShaderTemplateInfo _default_shader_template {
        "DeflMatShader.cu", false, {}, {}, {}
    };

    ShaderTemplateInfo _volume_shader_template {
        "volume.cu", false, {}, {}, {}
    };

    std::set<std::string> cachedMeshesMaterials, cachedSphereMaterials;

    void ensure_shadtmpl(ShaderTemplateInfo &_template) 
    {
        if (_template.ensured) return;

        _template.shadtmpl = sutil::lookupIncFile(_template.name.c_str());

        auto marker = std::string("//PLACEHOLDER");
        auto marker_length = marker.length();

        auto start_marker = _template.shadtmpl.find(marker);

        if (start_marker != std::string::npos) {
            auto end_marker = _template.shadtmpl.find(marker, start_marker + marker_length);

            _template.shadtmpl.replace(start_marker, marker_length, "/*PLACEHOLDER");
            _template.shadtmpl.replace(end_marker, marker_length, "PLACEHOLDER*/");
        }

        std::string_view tplsv = _template.shadtmpl;
        std::string_view tmpcommon = "//COMMON_CODE";
        auto pcommon = tplsv.find(tmpcommon);
        auto pcomend = pcommon;
        if(pcommon != std::string::npos)
        {
            pcomend = pcommon + tmpcommon.size();
            _template.commontpl = tplsv.substr(0, pcommon);
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
                _template.shadtpl2 = {tplsv.substr(pcomend, p0-pcomend), tplsv.substr(q1)};
            } else {
                throw std::runtime_error("cannot find stub GENERATED_END_MARK in shader template");
            }
        } else {
            throw std::runtime_error("cannot find stub GENERATED_BEGIN_MARK in shader template");
        }
        
        _template.ensured = true;
    }

    bool hasEnding (std::string const &fullString, std::string const &ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }

    void draw() override {
        //std::cout<<"in draw()"<<std::endl;
#ifdef OPTIX_BASE_GL
        auto guard = setupState();
#endif
        auto const &cam = *scene->camera;
        auto const &opt = *scene->drawOptions;

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
        if(scene->drawOptions->needRefresh){
            camNeedUpdate = true;
            scene->drawOptions->needRefresh = false;
        }

        //std::cout << "Render Options: SimpleRender " << scene->drawOptions->simpleRender
        //          << " NeedRefresh " << scene->drawOptions->needRefresh << "\n";

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
        //xinxinoptix::setAperature(cam.m_aperture);
        auto lodright = glm::normalize(glm::cross(cam.m_lodfront, cam.m_lodup));
        auto lodup = glm::normalize(glm::cross(lodright, cam.m_lodfront));
        //zeno::log_warn("lodup = {}", zeno::other_to_vec<3>(cam.m_lodup));
        //zeno::log_warn("lodfront = {}", zeno::other_to_vec<3>(cam.m_lodfront));
        //zeno::log_warn("lodright = {}", zeno::other_to_vec<3>(lodright));
        xinxinoptix::set_perspective(glm::value_ptr(lodright), glm::value_ptr(lodup),
                                     glm::value_ptr(cam.m_lodfront), glm::value_ptr(cam.m_lodcenter),
                                     cam.getAspect(), cam.m_fov, cam.focalPlaneDistance, cam.m_aperture);
        //xinxinoptix::set_projection(glm::value_ptr(cam.m_proj));
        }
        if(lightNeedUpdate){
            //zeno::log_debug("[zeno-optix] updating light");
            xinxinoptix::optixupdatelight();

            lightNeedUpdate = false;
        }

        if (meshNeedUpdate || matNeedUpdate || staticNeedUpdate) {
            //zeno::log_debug("[zeno-optix] updating scene");
            //zeno::log_debug("[zeno-optix] updating material");
            std::vector<ShaderPrepared> _mesh_shader_list{};
            std::vector<ShaderPrepared> _sphere_shader_list{};
            std::vector<ShaderPrepared> _volume_shader_list{};

            std::map<std::string, int> meshMatLUT{};
            std::map<std::string, uint> matIDtoShaderIndex{};

            ensure_shadtmpl(_default_shader_template);
            ensure_shadtmpl(_volume_shader_template);

            _mesh_shader_list.push_back({
                ShaderMaker::Mesh,
                "Default",
                _default_shader_template.shadtmpl,
                std::vector<std::string>()
            });

            meshMatLUT.clear();
            meshMatLUT.insert({"Default", 0});

            OptixUtil::g_vdb_indice_visible.clear();
            OptixUtil::g_vdb_list_for_each_shader.clear();

            //first pass, remove duplicated mat and keep the later
            std::map<std::string, GraphicsManager::DetMaterial*> matMap;
            std::map<std::string, int> order;
            for (auto const &[key, obj]: graphicsMan->graphics){
                if (auto mtldet = std::get_if<GraphicsManager::DetMaterial>(&obj->det)) {
                    auto matkey = mtldet->mtlidkey;
                    if(matMap.find(matkey)!=matMap.end())
                    {
                        auto idx = order[matkey];
                        auto curr_idx = graphicsMan->objOrder[key];
                        if(curr_idx < idx) {
                            matMap[matkey] = mtldet;
                            order[matkey] = curr_idx;
                        }
                    }
                    else
                    {
                        matMap.insert({matkey, mtldet});
                        auto curr_idx = graphicsMan->objOrder[key];
                        order[matkey] = curr_idx;
                    }
                }
            }

            if ( matNeedUpdate && (staticNeedUpdate || meshNeedUpdate) ) {
                cachedMeshesMaterials = xinxinoptix::uniqueMatsForMesh();
                cachedSphereMaterials = xinxinoptix::uniqueMatsForSphere();
            } // preserve material names for materials-only updating case 

            //for (auto const &[key, obj]: graphicsMan->graphics)
            for(auto const &[matkey, mtldet] : matMap)
            {
                    bool has_vdb = false;
                    if (mtldet->tex3Ds.size() > 0) {
                        glm::f64mat4 linear_transform(1.0);  
                        prepareVolumeTransform(mtldet->transform, linear_transform);
                        
                        std::vector<std::string> g_vdb_list_for_this_shader;
                        g_vdb_list_for_this_shader.reserve(mtldet->tex3Ds.size());

                        for (uint k=0; k<mtldet->tex3Ds.size(); ++k) 
                        {
                            auto& tex = mtldet->tex3Ds.at(k);
                            auto vdb_path = tex->path;

                            static const auto extension = std::string("vdb");
                            auto found_vdb = hasEnding(vdb_path, extension);
                            if (!found_vdb) { continue; }

                            auto index_of_shader = _volume_shader_list.size();
                            std::string combined_key;

                            auto loaded = OptixUtil::preloadVDB(*tex, index_of_shader, k, linear_transform, combined_key); 
                            has_vdb = has_vdb || loaded;

                            g_vdb_list_for_this_shader.push_back(combined_key);
                        }
                        if (has_vdb) {
                            OptixUtil::g_vdb_list_for_each_shader[_volume_shader_list.size()] = (g_vdb_list_for_this_shader);
                        }
                    }
                    
                    auto selected_template = has_vdb? _volume_shader_template : _default_shader_template; 

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

                    auto& commontpl = selected_template.commontpl;
                    auto& shadtpl2 = selected_template.shadtpl2;

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

                    std::vector<std::string> shaderTex;
                    int texid=0;
                    for(auto tex:mtldet->tex2Ds)
                    {
                        OptixUtil::addTexture(tex->path.c_str());
                        shaderTex.emplace_back(tex->path);
                        texid++;
                    }

                    ShaderPrepared shaderP; 
                        shaderP.material = mtldet->mtlidkey;
                        shaderP.source = shader;
                        shaderP.tex_names = shaderTex;

                    if (has_vdb) {
                         
                        shaderP.mark = ShaderMaker::Volume;
                        _volume_shader_list.push_back(shaderP);
                    } else {

                        if (cachedMeshesMaterials.count(mtldet->mtlidkey) > 0) {
                          meshMatLUT.insert(
                              {mtldet->mtlidkey, (int)_mesh_shader_list.size()});

                          shaderP.mark = ShaderMaker::Mesh;
                          _mesh_shader_list.push_back(shaderP);
                        }

                        if (cachedSphereMaterials.count(mtldet->mtlidkey) > 0) {

                            shaderP.mark = ShaderMaker::Sphere;
                            _sphere_shader_list.push_back(shaderP);
                        }
                    }
            }

            std::vector<ShaderPrepared> allShaders{};
            allShaders.reserve(_mesh_shader_list.size()+_sphere_shader_list.size()+_volume_shader_list.size());            

            allShaders.insert(allShaders.end(), _mesh_shader_list.begin(), _mesh_shader_list.end());
            allShaders.insert(allShaders.end(), _sphere_shader_list.begin(), _sphere_shader_list.end());
            allShaders.insert(allShaders.end(), _volume_shader_list.begin(), _volume_shader_list.end());

            const size_t sphere_shader_offset = _mesh_shader_list.size();
            const size_t volume_shader_offset = _mesh_shader_list.size() + _sphere_shader_list.size();

                for (uint i=0; i<allShaders.size(); ++i) {
                    auto& ref = allShaders[i];

                    auto combinedID = ref.material + ":" + std::to_string((ref.mark));
                    matIDtoShaderIndex[combinedID] = i;
                }

            if (matNeedUpdate)
            {
                std::cout<<"shaders size "<< allShaders.size() << std::endl;
                xinxinoptix::optixupdatematerial(allShaders);
                xinxinoptix::updateVolume(volume_shader_offset);
            }

            xinxinoptix::foreach_sphere_crowded([&matIDtoShaderIndex, sphere_shader_offset](const std::string &mtlid, std::vector<uint> &sbtoffset_list) {

                auto combinedID = mtlid + ":" + std::to_string(ShaderMaker::Sphere);

                if (matIDtoShaderIndex.count(combinedID) > 0) {
                    auto shaderIndex = matIDtoShaderIndex.at(combinedID);
                    sbtoffset_list.push_back(shaderIndex - sphere_shader_offset);
                }
            });

            xinxinoptix::SpheresCrowded.sbt_count = _sphere_shader_list.size();
            OptixUtil::matIDtoShaderIndex = matIDtoShaderIndex;

            if (meshNeedUpdate) 
            {
            
            OptixUtil::logInfoVRAM("Before update Mesh");

                if(staticNeedUpdate)
                    xinxinoptix::UpdateStaticMesh(meshMatLUT);

                xinxinoptix::UpdateDynamicMesh(meshMatLUT);

            OptixUtil::logInfoVRAM("Before update Inst");
                xinxinoptix::UpdateInst();
            OptixUtil::logInfoVRAM("After update Inst");

                xinxinoptix::updateInstancedSpheresGAS();
                xinxinoptix::updateCrowdedSpheresGAS();
                xinxinoptix::updateUniformSphereGAS();

            OptixUtil::logInfoVRAM("After update Sphere");

                xinxinoptix::UpdateStaticInstMesh(meshMatLUT);
                xinxinoptix::UpdateDynamicInstMesh(meshMatLUT);
                xinxinoptix::CopyInstMeshToGlobalMesh();
                xinxinoptix::UpdateGasAndIas(staticNeedUpdate);
            }
        
            xinxinoptix::optixupdateend();
            std::cout<<"optix update End\n";
            xinxinoptix::cleanupSpheres();
            std::cout<<"cleanupSpheres\n";

            matNeedUpdate = false;
            meshNeedUpdate = false;
            staticNeedUpdate = false;
        }

#ifdef OPTIX_BASE_GL
        int targetFBO = 0;
        CHECK_GL(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &targetFBO));
        {
            auto bindVao = opengl::scopeGLBindVertexArray(vao->vao);
            xinxinoptix::optixrender(targetFBO, scene->drawOptions->num_samples, scene->drawOptions->denoise, scene->drawOptions->simpleRender);
        }
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, targetFBO));
#else
        xinxinoptix::optixrender(0, scene->drawOptions->num_samples, scene->drawOptions->denoise, scene->drawOptions->simpleRender);
#endif
    }

    ~RenderEngineOptx() override {
        xinxinoptix::optixcleanup();
    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineOptx>("optx");

} // namespace zenovis::optx
#endif
