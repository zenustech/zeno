#include <tuple>
#include <unordered_map>
#include <vcruntime_string.h>
#include <vector_types.h>
#ifdef ZENO_ENABLE_OPTIX

#include "Scene.h"
#include "optixPathTracer.h"
#include "vec_math.h"
#include "xinxinoptixapi.h"
#include "zeno/utils/vec.h"
#include <limits>
#include <memory>
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
#include <zeno/types/AttrVector.h>
#include <tinygltf/json.hpp>

#include <map>
#include <string>
#include <string_view>
#include <random>

#include <hair/Hair.h>
#include <hair/optixHair.h>

#include "ShaderBuffer.h"
#include <zeno/extra/ShaderNode.h>

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
static void cleanMesh(zeno::PrimitiveObject* prim,
               std::vector<zeno::vec3f> &verts,
               std::vector<zeno::vec3f> &nrm,
               std::vector<zeno::vec3f> &clr,
               std::vector<zeno::vec3f> &tang,
               std::vector<zeno::vec3f> &uv,
               std::vector<zeno::vec3i> &idxBuffer)
{
    if(prim->has_attr("clr")==false)
    {
        prim->verts.add_attr<zeno::vec3f>("clr");
    }
  //first pass, scan the prim to see if verts require duplication
  std::vector<std::vector<zeno::vec3f>> vert_uv;
  std::vector<std::vector<zeno::vec2i>> idx_mapping;
  vert_uv.resize(prim->verts.size());
  idx_mapping.resize(prim->verts.size());
  int count = 0;
  for(int i=0;i<prim->tris.size();i++)
  {
    //so far, all value has already averaged on verts, except uv
    zeno::vec3i idx = prim->tris[i];
    for(int j=0;j<3;j++)
    {
      std::string uv_name;
      uv_name = "uv" + std::to_string(j);
      auto vid = idx[j];
      if(vert_uv[vid].size()==0)
      {
        vert_uv[vid].push_back(prim->tris.attr<zeno::vec3f>(uv_name)[i]);
        //idx_mapping[vid].push_back(zeno::vec2i(vid,count));
        //count++;
      }
      else
      {
        zeno::vec3f uv = prim->tris.attr<zeno::vec3f>(uv_name)[i];
        bool have = false;
        for(int k=0;k<vert_uv[vid].size();k++)
        {
          auto & tester = vert_uv[vid][k];
          if(tester[0] == uv[0] && tester[1] == uv[1] && tester[2] == uv[2] )
          {
            have = true;
          }
        }
        if(have == false)
        {
          //need a push_back
          vert_uv[vid].push_back(prim->tris.attr<zeno::vec3f>(uv_name)[i]);
          //idx_mapping[vid].push_back(zeno::vec2i(vid,count));
          //count++;
        }
      }
    }
  }
  count = 0;
  for(int i=0;i<vert_uv.size();i++) {
    for(int j=0;j<vert_uv[i].size();j++) {
      idx_mapping[i].push_back(zeno::vec2i(i, count));
      count++;
    }
  }
  //first pass done

  // [old_idx, new_idx ] = idx_mapping[vid][k] tells index mapping of old and new vert

  //run a pass to assemble new data
  verts.resize(0);
  nrm.resize(0);
  clr.resize(0);
  uv.resize(0);
  tang.resize(0);
  verts.reserve(count);
  nrm.reserve(count);
  clr.reserve(count);
  uv.reserve(count);
  tang.reserve(count);
  for(int i=0;i<vert_uv.size();i++)
  {
    for(int j=0;j<vert_uv[i].size();j++)
    {
      auto vid = idx_mapping[i][j][0];
      auto uvt = vert_uv[i][j];
      auto v  = prim->verts[vid];
      auto n  = prim->verts.attr<zeno::vec3f>("nrm")[vid];
      auto c  = prim->verts.attr<zeno::vec3f>("clr")[vid];
      auto t  = prim->verts.attr<zeno::vec3f>("atang")[vid];
      verts.push_back(v);
      nrm.push_back(n);
      clr.push_back(c);
      tang.push_back(t);
      uv.push_back(uvt);
    }
  }

  idxBuffer.resize(prim->tris.size());
  //third pass: assemble new idx map
  for(int i=0;i<prim->tris.size();i++)
  {
    zeno::vec3i idx = prim->tris[i];
    for(int j=0;j<3;j++) {

      auto old_vid = idx[j];
      if(idx_mapping[old_vid].size()==1)
      {
        idxBuffer[i][j] = idx_mapping[old_vid][0][1];
      }
      else
      {
        std::string uv_name = "uv" + std::to_string(j);
        auto &tuv = prim->tris.attr<zeno::vec3f>(uv_name)[i];
        for(int k=0;k<vert_uv[old_vid].size();k++)
        {
          auto &vuv = vert_uv[old_vid][k];
          if(vuv[0] == tuv[0] && vuv[1] == tuv[1] && vuv[2] == tuv[2])
          {
            idxBuffer[i][j] = idx_mapping[old_vid][k][1];
          }
        }
      }
    }
  }
}
struct GraphicsManager {
    Scene *scene;

        struct DetMaterial {
            std::vector<std::shared_ptr<zeno::Texture2DObject>> tex2Ds;
            std::vector<std::shared_ptr<zeno::TextureObjectVDB>> tex3Ds;
            std::string common;
            std::string shader;
            std::string extensions;
            std::string mtlidkey;
            std::string parameters;

            int stamp_base = 0;
            bool dirty = true;
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
            for(auto i=0;i<atang.size();i++)
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
                for (auto i = 0; i < prim->tris.size(); ++i) {
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
                for (auto i = 0; i < prim->tris.size(); ++i) {
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
            const auto stamp_work = [](const zeno::UserData& ud) {

                auto stamp_base = ud.get2<int>("stamp-base", 0);
                auto stamp_change = ud.get2<std::string>("stamp-change", "TotalChange");
                std::transform(stamp_change.begin(), stamp_change.end(), stamp_change.begin(), ::tolower);

                return std::tuple{stamp_base, stamp_change};
            };

            if (auto const *prim_in0 = dynamic_cast<zeno::PrimitiveObject *>(obj))
            {
                // vvv deepcopy to cihou following inplace ops vvv
                auto prim_in_lslislSp = std::make_shared<zeno::PrimitiveObject>(*prim_in0);
                // ^^^ Don't wuhui, I mean: Literial Synthetic Lazy internal static Local Shared Pointer
                auto prim_in = prim_in_lslislSp.get();

                const auto [stamp_base, stamp_change] = stamp_work(prim_in_lslislSp->userData());
                if (stamp_change == "unchanged") { return; }

                if ( prim_in->userData().has("ShaderAttributes") ) {
                    auto attritbutes  = prim_in->userData().get2<std::string>("ShaderAttributes");

                    using VarType = zeno::AttrVector<zeno::vec3f>::AttrVectorVariant;

                    auto json = nlohmann::json::parse(attritbutes);

                    for (auto& [attrName, bufferName] : json.items()) {
                        //for (auto& kname : keys) {
                        auto& val = prim_in->verts.attrs[attrName];
                        
                        std::visit([&, &bufferName=bufferName](auto&& arg) {
                            using T = std::decay_t<decltype(arg)>;

                            constexpr auto vsize = std::variant_size_v<VarType>;

                            zeno::static_for<0, vsize>([&, &bufferName=bufferName] (auto i) {
                                using ThisType = std::variant_alternative_t<i, VarType>;
                                using EleType = typename ThisType::value_type;

                                if constexpr (std::is_same_v<T, ThisType>) {

                                    auto& obj = reinterpret_cast<ThisType&>(val);

                                    if (obj.size() > 0) {

                                        size_t byte_size = obj.size() * sizeof(EleType);
                                        auto tmp_ptr = std::make_shared<xinxinoptix::raii<CUdeviceptr>>();
                                        tmp_ptr->resize(byte_size);
                                        cudaMemcpy((void*)tmp_ptr->handle, obj.data(), byte_size, cudaMemcpyHostToDevice);

                                        load_buffer_group(bufferName, tmp_ptr);
                                    }
                                    return true;
                                }
                                return false;
                            });

                        }, val);
                    }
                }

                if (prim_in->userData().has("curve") && prim_in->verts->size() && prim_in->verts.has_attr("width")) {

                    auto& ud = prim_in->userData();
                    auto mtlid = ud.get2<std::string>("mtlid", "Default");
                    auto curveTypeIndex = ud.get2<uint>("curve", 0u);
                    auto curveTypeEnum = magic_enum::enum_cast<zeno::CurveType>(curveTypeIndex).value_or(zeno::CurveType::LINEAR);

                    auto& widthArray = prim_in->verts.attr("width");
                    auto& pointArray = prim_in->verts;

                    std::vector<float3> dummy {};
                    
                    auto& normals = prim_in->verts.has_attr("v") ? reinterpret_cast<std::vector<float3>&>(prim_in->verts.attr("v")) : dummy;
                    auto& points = reinterpret_cast<std::vector<float3>&>(pointArray);
                    auto& widths = reinterpret_cast<std::vector<float>&>(widthArray);

                    std::vector<uint> strands {};

                    int begin = 0;
                    int end = 1;

                    if (prim_in->lines[0][1] < prim_in->lines[0][0]) {
                        std::swap(begin, end);
                    }

                    strands.push_back(prim_in->lines[0][begin]);

                    for (size_t i=1; i<prim_in->lines->size(); ++i) {
                        auto& prev_segment = prim_in->lines[i-1];
                        auto& this_segment = prim_in->lines[i];

                        if (prev_segment[end] != this_segment[begin]) { // new strand
                            strands.push_back(this_segment[begin]);
                        }
                    }

                    auto abcpath = ud.get2<std::string>("abcpath_0", "Default");
                    loadCurveGroup(points, widths, normals, strands, curveTypeEnum, mtlid, abcpath);
                    return;
                }

                auto is_cyhair = prim_in_lslislSp->userData().has("cyhair");
                if (is_cyhair) {
                    auto& ud = prim_in_lslislSp->userData();
                    auto mtlid = ud.get2<std::string>("mtlid", "Default");

                    auto type_index = ud.get2<uint>("curve", 0u);
                    auto path_string = ud.get2<std::string>("path", "");

                    glm::mat4 transform(1.0f);
                    auto transform_ptr = glm::value_ptr(transform);

                    if (ud.has("_transform_row0") && ud.has("_transform_row1") && ud.has("_transform_row2") && ud.has("_transform_row3")) {

                        auto row0 = ud.get2<zeno::vec4f>("_transform_row0");
                        auto row1 = ud.get2<zeno::vec4f>("_transform_row1");
                        auto row2 = ud.get2<zeno::vec4f>("_transform_row2");
                        auto row3 = ud.get2<zeno::vec4f>("_transform_row3");

                        memcpy(transform_ptr, row0.data(), sizeof(float)*4);
                        memcpy(transform_ptr+4, row1.data(), sizeof(float)*4);
                        memcpy(transform_ptr+8, row2.data(), sizeof(float)*4);  
                        memcpy(transform_ptr+12, row3.data(), sizeof(float)*4); 
                    }

                    auto yup = ud.get2<bool>("yup", true);
                    auto trans = yup? glm::mat4 { 
                                                    0, 0, 1, 0,
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 0, 1
                                                } : glm::mat4(1.0);

                    trans = transform * trans;
                    loadHair( path_string, mtlid, type_index, trans);
                    return;
                }

                auto is_sphere = prim_in_lslislSp->userData().has("sphere_center");
                if (is_sphere) {

                    auto& ud = prim_in_lslislSp->userData();
                    printf("Before loading sphere %s for ray tracing... \n", key.c_str());
                    
                    auto mtlid = ud.get2<std::string>("mtlid", "Default");
                    auto instID = ud.get2<std::string>("instID", "Default");

                    bool instanced = (instID != "Default" && instID != "");

                    auto sphere_scale = ud.get2<zeno::vec3f>("sphere_scale");
                    auto uniform_scaling = sphere_scale[0] == sphere_scale[1] && sphere_scale[2] == sphere_scale[0];

                    if (instanced) { 
                        auto sphere_center = ud.get2<zeno::vec3f>("sphere_center");
                        auto sphere_radius = ud.get2<float>("sphere_radius");
                             sphere_radius *= fmaxf(fmaxf(sphere_scale[0], sphere_scale[1]), sphere_scale[2]);

                    } else {

                        const auto objectName = prim_in->userData().get2<std::string>("ObjectName", key); 

                        //zeno::vec4f row0, row1, row2, row3;
                        auto row0 = ud.get2<zeno::vec4f>("_transform_row0");
                        auto row1 = ud.get2<zeno::vec4f>("_transform_row1");
                        auto row2 = ud.get2<zeno::vec4f>("_transform_row2");
                        auto row3 = ud.get2<zeno::vec4f>("_transform_row3");

                        glm::mat4 sphere_transform;
                        auto transform_ptr = glm::value_ptr(sphere_transform);

                        memcpy(transform_ptr, row0.data(), sizeof(float)*4);
                        memcpy(transform_ptr+4, row1.data(), sizeof(float)*4);
                        memcpy(transform_ptr+8, row2.data(), sizeof(float)*4);  
                        memcpy(transform_ptr+12, row3.data(), sizeof(float)*4);

                        defaultScene.preload_sphere_transformed(objectName, mtlid, instID, sphere_transform);
                    }

                    printf("After loading sphere %s for ray tracing... \n", key.c_str());
                    return;
                }

                auto is_vbox = prim_in_lslislSp->userData().has("vbox");
                if (is_vbox) {
                    auto& ud = prim_in_lslislSp->userData();
                    auto mtlid = ud.get2<std::string>("mtlid", "Default");

                    auto row0 = ud.get2<zeno::vec4f>("_transform_row0");
                    auto row1 = ud.get2<zeno::vec4f>("_transform_row1");
                    auto row2 = ud.get2<zeno::vec4f>("_transform_row2");
                    auto row3 = ud.get2<zeno::vec4f>("_transform_row3");

                    glm::mat4 vbox_transform;
                    auto transform_ptr = glm::value_ptr(vbox_transform);

                    memcpy(transform_ptr, row0.data(), sizeof(float)*4);
                    memcpy(transform_ptr+4, row1.data(), sizeof(float)*4);
                    memcpy(transform_ptr+8, row2.data(), sizeof(float)*4);  
                    memcpy(transform_ptr+12, row3.data(), sizeof(float)*4);

                    auto bounds = ud.get2<std::string>("bounds");
                    
                    uint8_t boundsID = [&]() {
                        if ("Box" == bounds)
                            return 0;
                        if ("Sphere" == bounds)
                            return 1;
                        if ("HemiSphere" == bounds)
                            return 2;
                    } ();

                    const auto reName = prim_in->userData().get2<std::string>("ObjectName", key);
                    defaultScene.preloadVolumeBox(reName, mtlid, boundsID, vbox_transform);
                    return;
                }

                auto isRealTimeObject = prim_in->userData().get2<int>("isRealTimeObject", 0);
                auto isUniformCarrier = prim_in->userData().has("ShaderUniforms");
                
                if (isRealTimeObject == 0 && isUniformCarrier == 0)
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

            if (prim_in->userData().has("ResourceType")) {
                    
                const auto reType = prim_in->userData().get2<std::string>("ResourceType", "Mesh");
                const auto reName = prim_in->userData().get2<std::string>("ObjectName", key);
                
                if (reType == "SceneDescriptor") 
                {
                    const auto sceneConfig = prim_in->userData().get2<std::string>("Scene", "");
                    defaultScene.preload_scene(sceneConfig);
                    return;
                }
            
                if (reType == "Matrixes") {
                    auto count = prim_in->verts->size();
                    auto& _r0 = prim_in->verts.attr<zeno::vec3f>("r0");
                    auto& _r1 = prim_in->verts.attr<zeno::vec3f>("r1");
                    auto& _r2 = prim_in->verts.attr<zeno::vec3f>("r2");

                    std::vector<m3r4c> matrix_list(count);

                    for (size_t i=0; i<count; ++i) {
                        auto pos = prim_in->verts[i];
                        auto r0 = _r0[i];
                        auto r1 = _r1[i];
                        auto r2 = _r2[i];

                        auto& matrix3r4c = matrix_list[i];
                        matrix3r4c[0]  = r0[0];
                        matrix3r4c[1]  = r1[0];
                        matrix3r4c[2]  = r2[0];
                        matrix3r4c[3]  = pos[0];
                        matrix3r4c[4]  = r0[1];
                        matrix3r4c[5]  = r1[1];
                        matrix3r4c[6]  = r2[1];
                        matrix3r4c[7]  = pos[1];
                        matrix3r4c[8]  = r0[2];
                        matrix3r4c[9]  = r1[2];
                        matrix3r4c[10] = r2[2];
                        matrix3r4c[11] = pos[2];
                    }
                    defaultScene.load_matrix_list(reName, matrix_list);
                    return;
                }

                if (reType == "Particles") {
                    auto& center = prim_in->verts;
                    auto& color = prim_in->verts.attr<zeno::vec3f>("clr");
                    auto& radius = prim_in->verts.attr<float>("radius");

                    defaultScene.preload_sphere_group(reName, center, radius, color);
                    return;
                }

                if (reType == "Mesh") 
                {

                    if (prim_in->quads.size() || prim_in->polys.size()) {
                        zeno::log_trace("demoting faces");
                        zeno::primTriangulateQuads(prim_in);
                        zeno::primTriangulate(prim_in);
                    }
                    if(prim_in->tris.size()==0) return;

                    bool has_uv =   prim_in->tris.has_attr("uv0")&&prim_in->tris.has_attr("uv1")&&prim_in->tris.has_attr("uv2");
                    if(has_uv == false)
                    {
                        prim_in->tris.add_attr<zeno::vec3f>("uv0");
                        prim_in->tris.add_attr<zeno::vec3f>("uv1");
                        prim_in->tris.add_attr<zeno::vec3f>("uv2");
                    }
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
                    
                    std::vector<zeno::vec3f> verts;
                    std::vector<zeno::vec3f> nrm;
                    std::vector<zeno::vec3f> clr;
                    std::vector<zeno::vec3f> tang;
                    std::vector<zeno::vec3f> uv;
                    std::vector<zeno::vec3i> idxBuffer;
                    cleanMesh(prim_in, verts, nrm, clr, tang, uv, idxBuffer);
                    auto oPrim = std::make_shared<zeno::PrimitiveObject>();
                    oPrim->verts.resize(verts.size());
                    oPrim->add_attr<zeno::vec3f>("nrm");
                    oPrim->add_attr<zeno::vec3f>("clr");
                    oPrim->add_attr<zeno::vec3f>("uv");
                    oPrim->add_attr<zeno::vec3f>("atang");
                    oPrim->tris.resize(idxBuffer.size());

                    oPrim->verts.attr<zeno::vec3f>("pos") = verts;
                    oPrim->verts.attr<zeno::vec3f>("nrm") = nrm;
                    oPrim->verts.attr<zeno::vec3f>("clr") = clr;
                    oPrim->verts.attr<zeno::vec3f>("uv") = uv;
                    oPrim->verts.attr<zeno::vec3f>("atang") = tang;
                    oPrim->tris = idxBuffer;

                    auto vs = (float const *)oPrim->verts.data();
                    std::map<std::string, std::pair<float const *, size_t>> vtab;
                    oPrim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
                        vtab[key] = {(float const *)arr.data(), sizeof(arr[0]) / sizeof(float)};
                    });
                    auto ts = (uint const *)oPrim->tris.data();
                    auto nvs = oPrim->verts.size();
                    auto nts = oPrim->tris.size();

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
                    auto mtlid = prim_in->userData().get2<std::string>("Material_0", "Default");
                    auto& matids = prim_in->tris.attr<int>("matid");

                    defaultScene.preload_mesh(reName, mtlid, vs, nvs, ts, nts, vtab, matids.data(), matNameList);
                } // Mesh
            } // ResourceType

                }
            }
            else if (auto mtl = dynamic_cast<zeno::MaterialObject *>(obj))
            {
                const auto [stamp_base, stamp_change] = stamp_work(mtl->userData());
                const auto dirty = stamp_change != "unchanged";
                det = DetMaterial{mtl->tex2Ds, mtl->tex3Ds, mtl->common, mtl->frag, mtl->extensions, mtl->mtlidkey, mtl->parameters,
                        stamp_base,  dirty};
            }
        }

        ~ZxxGraphic() {
            defaultScene.unload_object(key);
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
                auto type = prim_in->userData().get2<int>("type", 0);
                auto shape = prim_in->userData().get2<int>("shape", 0);
                auto maxDistance = prim_in->userData().get2<float>("maxDistance", std::numeric_limits<float>().max());
                auto falloffExponent = prim_in->userData().get2<float>("falloffExponent", 2.0f);

                auto color = prim_in->userData().get2<zeno::vec3f>("color");
                auto spread = prim_in->userData().get2<zeno::vec2f>("spread", {1.0f, 0.0f});
                auto intensity = prim_in->userData().get2<float>("intensity", 1.0f);
                auto fluxFixed = prim_in->userData().get2<float>("fluxFixed", -1.0f);
                auto vIntensity = prim_in->userData().get2<float>("visibleIntensity", -1.0f);

                auto ivD = prim_in->userData().getLiterial<int>("ivD", 0);

                auto mask = prim_in->userData().get2<int>("mask", 255);
                auto visible = prim_in->userData().get2<int>("visible", 0);
                auto doubleside = prim_in->userData().get2<int>("doubleside", 0);
                auto lightProfilePath = prim_in->userData().get2<std::string>("lightProfile", ""); 
                auto lightTexturePath = prim_in->userData().get2<std::string>("lightTexture", ""); 
                auto lightGamma = prim_in->userData().get2<float>("lightGamma", 1.0f); 

                if (lightProfilePath != "") {
                    OptixUtil::addTexture(lightProfilePath);
                }

                if (lightTexturePath != "") {
                    OptixUtil::addTexture(lightTexturePath);
                }

                xinxinoptix::LightDat ld;
                zeno::vec3f nor{}, clr{};

                ld.mask = mask;
                ld.visible = visible;
                ld.doubleside = doubleside;
                ld.fluxFixed = fluxFixed;
                ld.intensity = intensity;
                ld.vIntensity = vIntensity;
                ld.spreadMajor = spread[0];
                ld.spreadMinor = spread[1];
                ld.maxDistance = maxDistance;
                ld.falloffExponent = falloffExponent;

                ld.shape = shape; ld.type = type;
                ld.profileKey = lightProfilePath;
                ld.textureKey = lightTexturePath;
                ld.textureGamma = lightGamma;

                std::function extraStep = [&]() {
                    ld.normal.assign(nor.begin(), nor.end());
                    ld.color.assign(clr.begin(), clr.end());
                };

                const auto shapeEnum = magic_enum::enum_cast<zeno::LightShape>(shape);
                if (shapeEnum == zeno::LightShape::TriangleMesh) { // Triangle mesh Light

                    for (size_t i=0; i<prim_in->tris->size(); ++i) {
                        auto _p0_ = prim_in->verts[prim_in->tris[i][0]];
                        auto _p1_ = prim_in->verts[prim_in->tris[i][1]];
                        auto _p2_ = prim_in->verts[prim_in->tris[i][2]];
                        auto _e1_ = _p0_ - _p2_;
                        auto _e2_ = _p1_ - _p2_;

                        zeno::vec3f *pn0{}, *pn1{}, *pn2{};
                        zeno::vec3f *uv0{}, *uv1{}, *uv2{};

                        if (prim_in->verts.has_attr("nrm")) {
                            pn0 = &prim_in->verts.attr<zeno::vec3f>("nrm")[ prim_in->tris[i][0] ];
                            pn1 = &prim_in->verts.attr<zeno::vec3f>("nrm")[ prim_in->tris[i][1] ];
                            pn2 = &prim_in->verts.attr<zeno::vec3f>("nrm")[ prim_in->tris[i][2] ];
                        }

                        if (prim_in->verts.has_attr("uv")) {
                            #if 1
                            uv0 = &prim_in->verts.attr<zeno::vec3f>("uv")[ prim_in->tris[i][0] ];
                            uv1 = &prim_in->verts.attr<zeno::vec3f>("uv")[ prim_in->tris[i][1] ];
                            uv2 = &prim_in->verts.attr<zeno::vec3f>("uv")[ prim_in->tris[i][2] ];
                            #else
                            auto uv0 = prim_in->tris.attr<zeno::vec3f>("uv0")[i];
                            auto uv1 = prim_in->tris.attr<zeno::vec3f>("uv1")[i];
                            auto uv2 = prim_in->tris.attr<zeno::vec3f>("uv2")[i];
                            #endif
                        }

                        nor = zeno::normalize(zeno::cross(_e1_, _e2_));
                        clr = color ;//prim_in->verts.attr<zeno::vec3f>("clr")[ prim_in->tris[i][0] ];
                        extraStep();

                        auto compound = key + std::to_string(i);
                        xinxinoptix::load_triangle_light(compound, ld, _p0_, _p1_, _p2_, pn0, pn1, pn2, uv0, uv1, uv2); 
                    }
                } 
                else 
                {
                    auto v0 = prim_in->verts[0];
                    auto v1 = prim_in->verts[1];
                    auto v3 = prim_in->verts[3];
                    auto e1 = v1 - v3;
                    auto e2 = v0 - v1;
                    
                    // v3 ---(+x)--> v1
                    // |||||||||||||(-)
                    // |||||||||||||(z)
                    // |||||||||||||(+)
                    // v2 <--(-x)--- v0

                    v3 = v3 + e2; // p* as p0
                    e2 = -e2;     // invert e2
                
                    // facing down in local space
                    auto ne2 = zeno::normalize(e2);
                    auto ne1 = zeno::normalize(e1);
                    nor = zeno::normalize(zeno::cross(ne2, ne1));
                    //if (ivD) { nor *= -1; }

                    if (prim_in->verts.has_attr("clr")) {
                        clr = prim_in->verts.attr<zeno::vec3f>("clr")[0];
                    } else {
                        clr = zeno::vec3f(30000.0f, 30000.0f, 30000.0f);
                    }

                    clr = color;
                    extraStep();

                    std::cout << "light: v"<<v3[0]<<" "<<v3[1]<<" "<<v3[2]<<"\n";
                    std::cout << "light: v"<<v1[0]<<" "<<v1[1]<<" "<<v1[2]<<"\n";
                    std::cout << "light: v"<<v0[0]<<" "<<v0[1]<<" "<<v0[2]<<"\n";
                    std::cout << "light: e"<<e1[0]<<" "<<e1[1]<<" "<<e1[2]<<"\n";
                    std::cout << "light: e"<<e2[0]<<" "<<e2[1]<<" "<<e2[2]<<"\n";
                    std::cout << "light: n"<<nor[0]<<" "<<nor[1]<<" "<<nor[2]<<"\n";
                    std::cout << "light: c"<<clr[0]<<" "<<clr[1]<<" "<<clr[2]<<"\n";

                    xinxinoptix::load_light(key, ld, v3.data(), e1.data(), e2.data());
                }
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
                if (!path.empty()) {
                    if (OptixUtil::sky_tex.has_value() && OptixUtil::sky_tex.value() != path
                        && OptixUtil::sky_tex.value() != OptixUtil::default_sky_tex ) {
                        OptixUtil::removeTexture( {OptixUtil::sky_tex.value(), false} );
                    }

                    OptixUtil::sky_tex = path;
                    OptixUtil::addSkyTexture(path);
                } else {
                    OptixUtil::sky_tex = OptixUtil::default_sky_tex;
                }

                xinxinoptix::update_hdr_sky(evnTexRotation, evnTex3DRotation, evnTexStrength);
                xinxinoptix::using_hdr_sky(enableHdr);

                if (OptixUtil::portal_delayed.has_value()) {
                    OptixUtil::portal_delayed.value()();
                    //OptixUtil::portal_delayed.reset();
                }
            }
            else if (prim_in->userData().has<int>("SkyComposer")) {

                auto& attr_dir = prim_in->verts;

                std::vector<zeno::DistantLightData> dlights;
                dlights.reserve(attr_dir->size());
                
                if (attr_dir->size()) {

                    auto& attr_angle = attr_dir.attr<float>("angle");
                    auto& attr_color = attr_dir.attr<zeno::vec3f>("color");
                    auto& attr_inten = attr_dir.attr<float>("inten");

                    for (size_t i=0; i<attr_dir->size(); ++i) {

                        auto& dld = dlights.emplace_back();
                        dld.direction = attr_dir[i];
                        dld.angle = attr_angle[i];
                        dld.color = attr_color[i];
                        dld.intensity = attr_inten[i];
                    }
                }
                xinxinoptix::updateDistantLights(dlights);

                if(prim_in->userData().has<std::string>("portals")) {

                    auto portals_string = prim_in->userData().get2<std::string>("portals");
                    auto portals_json = nlohmann::json::parse(portals_string);

                    auto ps_string = prim_in->userData().get2<std::string>("psizes");
                    auto ps_json = nlohmann::json::parse(ps_string);

                    std::vector<Portal> portals {};

                    if (portals_json.is_array() && portals_json.size()%4 == 0) {
                        
                        portals.reserve(portals_json.size()/4);

                        auto pack = [&portals_json](size_t i) {
                            auto x = portals_json[i][0].template get<float>();
                            auto y = portals_json[i][1].template get<float>();
                            auto z = portals_json[i][2].template get<float>();
                            return zeno::vec3f(x, y, z);
                        };
                        
                        for (size_t i=0; i<portals_json.size(); i+=4) {
                            auto v0 = pack(i+0);
                            auto v1 = pack(i+1);
                            auto v2 = pack(i+2);
                            auto v3 = pack(i+3);

                            uint32_t psize = ps_json[i/4].template get<int>(); 
                            portals.push_back({v0, v1, v2, v3, psize});
                        }
                    } 

                    if (OptixUtil::sky_tex.has_value()) {
                        xinxinoptix::updatePortalLights(portals);
                    }
                    OptixUtil::portal_delayed = [=]() {
                        xinxinoptix::updatePortalLights(portals);
                    }; 
                } //portals
            }
        }
        return sky_found;
    }

    bool need_update_light(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        auto ins = graphics.insertPass();

        bool changelight = false;
        for (auto const &[key, obj] : objs) {
            if(ins.may_emplace(key)) {
                const std::string& stamptag = obj->userData().get2<std::string>("stamp-change", "TotalChange");
                const int& stamptag1 = obj->userData().get2<int>("stamp-base", -1);
                const std::string& stamptag2 = obj->userData().get2<std::string>("stamp-dataChange-hint", "");
                changelight = true;
            }
        }
        {   //when turn off last node in always mode
            static int objsNum = 0;
            if (objsNum > objs.size() && !changelight)
                changelight = true;
            objsNum = objs.size();
            if (scene->drawOptions->updateMatlOnly)
                changelight = false;
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
                //zeno::log_info("load_static_object: loading graphics [{}]", key);
                changed = true;

                if (auto cam = dynamic_cast<zeno::CameraObject *>(obj))
                {
                    scene->camera->setCamera(cam->get());     // pyb fix
                    auto &ud = cam->userData();
                    if (ud.has("aces")) {
                        scene->camera->setPhysicalCamera(
                            ud.get2<float>("aperture"),
                            ud.get2<float>("shutter_speed"),
                            ud.get2<float>("iso"),
                            ud.get2<bool>("aces"),
                            ud.get2<bool>("exposure"),
                            ud.get2<bool>("panorama_camera"),
                            ud.get2<bool>("panorama_vr180"),
                            ud.get2<float>("pupillary_distance")
                        );
                    }
                }

                auto ig = std::make_unique<ZxxGraphic>(key, obj);

                //zeno::log_info("load_static_object: loaded graphics to {}", ig.get());
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

//                zeno::log_info("load_object: loading graphics [{}]", key);
                changed = true;

                if (!scene->drawOptions->updateMatlOnly) {
                    if (auto cam = dynamic_cast<zeno::CameraObject *>(obj)) {
                        scene->camera->setCamera(cam->get()); // pyb fix
                        auto &ud = cam->userData();
                        if (ud.has("aces")) {
                            scene->camera->setPhysicalCamera(
                                ud.get2<float>("aperture"),
                                ud.get2<float>("shutter_speed"),
                                ud.get2<float>("iso"),
                                ud.get2<bool>("aces"),
                                ud.get2<bool>("exposure"),
                                ud.get2<bool>("panorama_camera"),
                                ud.get2<bool>("panorama_vr180"),
                                ud.get2<float>("pupillary_distance")
                            );
                        }
                    }
                }

                if (0) {
                    auto& ud = obj->userData();
                    if (ud.has("stamp_mode")) {
                        std::string stamp_mode = ud.get2<std::string>("stamp_mode");
                        if (!stamp_mode.empty()) {
                        }
                    }
                }

                auto ig = std::make_unique<ZxxGraphic>(key, obj);

//                zeno::log_info("load_object: loaded graphics to {}", ig.get());
                ins.try_emplace(key, std::move(ig));
            }
        }
        {   //when turn off last node in always mode
            static int objsNum = 0;
            if (objsNum > objs.size() && !changed)
                changed = true;
            objsNum = objs.size();
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
    std::optional<glm::vec3> getClickedPos(int x, int y) override {
        glm::vec3 posWS = xinxinoptix::get_click_pos(x, y);
        if (posWS == glm::vec3()) {
            return {};
        }
        auto const &cam = *scene->camera;
        posWS += cam.m_pos;
        return posWS;
    }

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

#define MY_CAM_ID(cam) cam.m_nx, cam.m_ny, cam.m_rotation, cam.m_pos, cam.m_fov, cam.focalPlaneDistance, cam.m_aperture
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

    ShaderTemplateInfo _default_callable_template {
        "CallableDefault.cu", false, {}, {}, {}
    };
    ShaderTemplateInfo _volume_callable_template {
        "CallableVolume.cu", false, {}, {}, {}
    };
    ShaderTemplateInfo _default_shader_template {
        "DeflMatShader.cu", false, {}, {}, {}
    };
    ShaderTemplateInfo _volume_shader_template {
        "volume.cu", false, {}, {}, {}
    };

    ShaderTemplateInfo _light_shader_template {
        "Light.cu", false, {}, {}, {}
    };

    std::map<std::string, std::vector<zeno::CurveType>> cachedCurvesMaterials;

    std::map<std::string, std::set<ShaderMark>> required_shader_names;
    std::map<shader_key_t, std::shared_ptr<ShaderPrepared>, ByShaderKey> cached_shaders{};

    std::map<std::string, uint16_t> cachedMeshMatLUT;
    bool meshMatLUTChanged(std::map<std::string, uint16_t>& newLUT) {
        bool changed = false;
        if (cachedMeshMatLUT.size() != newLUT.size()) {
            changed = true;
        }
        else {
            for (auto const& [matkey, matidx] : newLUT)
            {
                if (cachedMeshMatLUT.count(matkey) == 0)
                    changed = true;
                else if (cachedMeshMatLUT[matkey] != newLUT[matkey])
                    changed = true;
            }
        }
        return changed;
    }

    void ensure_shadtmpl(ShaderTemplateInfo &_template) 
    {
        if (_template.ensured) return;

        _template.shadtmpl = sutil::lookupIncFile(_template.name.c_str());

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
            return;
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

    void draw(bool _) override {
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


        if (sizeNeedUpdate) {
            zeno::log_debug("[zeno-optix] updating resolution");
            xinxinoptix::set_window_size(cam.m_nx, cam.m_ny);

        }

        if (sizeNeedUpdate || camNeedUpdate) {
            zeno::log_debug("[zeno-optix] updating camera");
            auto lodright = cam.m_rotation * glm::vec3(1, 0, 0);
            auto lodup = cam.m_rotation * glm::vec3(0, 1, 0);
            auto lodfront = cam.m_rotation * glm::vec3(0, 0, -1);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int32_t> dis(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());

            xinxinoptix::set_outside_random_number(dis(gen));
        
            xinxinoptix::set_perspective(glm::value_ptr(lodright), glm::value_ptr(lodup),
                                        glm::value_ptr(lodfront), glm::value_ptr(cam.m_pos),
                                        cam.getAspect(), cam.m_fov, cam.focalPlaneDistance, cam.m_aperture);
            xinxinoptix::set_physical_camera_param(
                cam.zOptixCameraSettingInfo.aperture,
                cam.zOptixCameraSettingInfo.shutter_speed,
                cam.zOptixCameraSettingInfo.iso,
                cam.zOptixCameraSettingInfo.aces,
                cam.zOptixCameraSettingInfo.exposure,
                cam.zOptixCameraSettingInfo.panorama_camera,
                cam.zOptixCameraSettingInfo.panorama_vr180,
                cam.zOptixCameraSettingInfo.pupillary_distance
            );
        }


        //zeno::log_info("meshNeedUpdate:{},matNeedUpdate:{},staticNeedUpdate:{},lightNeedUpdate:{}",meshNeedUpdate,matNeedUpdate,staticNeedUpdate,lightNeedUpdate);
        if (meshNeedUpdate || matNeedUpdate || staticNeedUpdate) {

            std::map<std::string,  uint16_t> meshMatLUT{};
            std::map<shader_key_t, uint16_t> ShaderKeyIndex{};

            ensure_shadtmpl(_default_callable_template);
            ensure_shadtmpl(_volume_callable_template);

            ensure_shadtmpl(_default_shader_template);
            ensure_shadtmpl(_volume_shader_template);
            ensure_shadtmpl(_light_shader_template);

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

            const auto make_default_shader = [&](ShaderMark mark) {
                auto default_shader = std::make_shared<ShaderPrepared>();
                default_shader->mark = mark;
                default_shader->matid = "Default";
                default_shader->filename = _default_shader_template.name;
                default_shader->callable = _default_callable_template.shadtmpl;
                return default_shader;
            };

            bool ShaderDirty = false;

            if ( matNeedUpdate && (staticNeedUpdate || meshNeedUpdate) ) {
                required_shader_names = defaultScene.prepareShaderSet();

                const auto shaderCount = cached_shaders.size();

                for (const auto& [key, value] : required_shader_names) {

                    bool use_default = matMap.count(key) == 0;                       

                    for (const auto& mark : value) {
                        auto shader_key = std::tuple {key, mark};
                        //auto default_key = std::tuple {"Default", mark};
                        
                        if (use_default) {
                            auto shader_ref = cached_shaders[shader_key];
                            if (shader_ref==nullptr || shader_ref->dirty)
                                cached_shaders[shader_key] = make_default_shader(mark);
                        } else {
                            auto& shader_ref = matMap.at(key);
                            if (shader_ref->dirty) {
                                cached_shaders[shader_key] = nullptr;
                                ShaderDirty = true;
                            } // dirty
                        }
                    }
                }

                ShaderDirty |= (cached_shaders.size() != shaderCount);

                // for (auto& [key, _] : hair_xxx_cache) 
                // {
                //     auto& [filePath, mode, mtid] = key;

                //     auto ctype = (zeno::CurveType)mode;

                //     if (cachedCurvesMaterials.count(mtid) > 0) {
                //         auto& ref = cachedCurvesMaterials.at(mtid);
                //         ref.push_back( ctype );
                //         continue;
                //     }
                //     cachedCurvesMaterials[mtid] = { ctype };
                // }

                // for (auto& [key, ele] : curveGroupCache) {

                //     auto ctype = ele->curveType;
                //     auto mtlid = ele->mtlid;

                //     if (cachedCurvesMaterials.count(mtlid) > 0) {
                //         auto& ref = cachedCurvesMaterials.at(mtlid);
                //         ref.push_back( ctype );
                //         continue;
                //     }
                //     cachedCurvesMaterials[mtlid] = { ctype };
                // }
 
            } // preserve material names for materials-only updating case

            unsigned int usesCurveTypeFlags = 0;
            auto mark_task = [&usesCurveTypeFlags](zeno::CurveType ele) {

                usesCurveTypeFlags |= CURVE_FLAG_MAP.at(ele);
                return CURVE_SHADER_MARK.at(ele);
            };

            if (cachedCurvesMaterials.count("Default") ) {

                auto& ref = required_shader_names.at("Default"); 

                for (auto& ele : ref) {

                    auto tmp = std::make_shared<ShaderPrepared>();
                    tmp->matid = "Default";
                    tmp->filename = _default_shader_template.name;
                    tmp->callable = _default_callable_template.shadtmpl;
                    tmp->mark = ele;
                }                
            }

            // Auto unload unused texure
            {
                std::set<OptixUtil::TexKey> realNeedTexPaths;
                for(auto const &[matkey, mtldet] : matMap) {
                    if (required_shader_names.count( mtldet->mtlidkey ) > 0) 
                    {
                        for(auto& tex: mtldet->tex2Ds) {
                            realNeedTexPaths.insert( {tex->path, tex->blockCompression} );
                        }
                    }
                    
                }
                // add light map
                for(auto const &[_, ld]: xinxinoptix::get_lightdats()) {
                    if (ld.textureKey.size()) {
                        realNeedTexPaths.insert( {ld.textureKey, false});
                    }
                }
                std::vector<OptixUtil::TexKey> needToRemoveTexPaths;
                for(auto const &[key, _]: OptixUtil::tex_lut) {

                    if (realNeedTexPaths.count(key) > 0) {
                        continue; 
                    }
                    if (OptixUtil::sky_tex.has_value() && key.path == OptixUtil::sky_tex.value()) {
                        continue;
                    }
                    if (key.path == OptixUtil::default_sky_tex) {
                        continue;
                    }
                    needToRemoveTexPaths.emplace_back(key);
                }
                for (const auto& need_remove_tex: needToRemoveTexPaths) {
                    OptixUtil::removeTexture(need_remove_tex);
                }
                for (const auto& realNeedTexKey: realNeedTexPaths) {
                    OptixUtil::addTexture(realNeedTexKey.path, realNeedTexKey.blockCompression);
                }
            }

            for(auto const &[matkey, mtldet] : matMap)
            {       
                if ( !mtldet->dirty ) continue;
                if ( required_shader_names.count(mtldet->mtlidkey) == 0 ) continue;
                mtldet->dirty = false;

                    const bool isVol = mtldet->parameters.find("vol") != std::string::npos;
                    
                    const auto& selected_source = isVol? _volume_shader_template : _default_shader_template;
                    const auto& selected_callable = isVol? _volume_callable_template : _default_callable_template; 

                    std::string callable;
                    auto common_code = mtldet->common;

                    auto& commontpl = selected_callable.commontpl;
                    auto& shadtpl2 = selected_callable.shadtpl2;

                    callable.reserve(commontpl.size()
                                    + common_code.size()
                                    + shadtpl2.first.size()
                                    + mtldet->shader.size()
                                    + shadtpl2.second.size());
                    callable.append(commontpl);
                    callable.append(common_code);
                    callable.append(shadtpl2.first);
                    callable.append(mtldet->shader);
                    callable.append(shadtpl2.second);
                    //std::cout<<callable<<std::endl;

                    ShaderPrepared shaderP; 

                        shaderP.callable = callable;
                        shaderP.filename = selected_source.name;
                        shaderP.parameters = mtldet->parameters;

                        shaderP.matid = mtldet->mtlidkey;
                        shaderP.tex_keys.reserve(mtldet->tex2Ds.size());

                        for(auto tex:mtldet->tex2Ds)
                        {
                            shaderP.tex_keys.push_back( {tex->path, tex->blockCompression} );
                        }

                        if (mtldet->tex3Ds.size() > 0) {

                            shaderP.vdb_keys.resize(mtldet->tex3Ds.size());
        
                            for (uint k=0; k<mtldet->tex3Ds.size(); ++k) 
                            {
                                auto& tex = mtldet->tex3Ds.at(k);
                                auto vdb_path = tex->path;
        
                                static const auto extension = std::string("vdb");
                                auto found_vdb = zeno::ends_with(vdb_path, extension);
                                if (!found_vdb) { continue; }
        
                                std::string vdb_key;
                                auto loaded = defaultScene.preloadVDB(*tex, vdb_key); 
                                shaderP.vdb_keys[k] = vdb_key;
                            }
                        }

                    if (isVol) {
                        
                        shaderP.mark = ShaderMark::Volume;
                        auto this_key = std::tuple{shaderP.matid, ShaderMark::Volume};
                        cached_shaders[this_key] = std::make_shared<ShaderPrepared>(shaderP);
                    } else {

                        auto reuiredSet = required_shader_names.at(mtldet->mtlidkey);
                        
                        for (auto& mark : reuiredSet) {
                            shaderP.mark = mark;

                            auto _shader_key = std::tuple{mtldet->mtlidkey, mark};                            
                            cached_shaders[_shader_key] = std::make_shared<ShaderPrepared>(shaderP);
                        }
                    }
            }

            bool requireTriangObj = false;
            bool requireSphereObj = false;
            bool requireVolumeObj = false;

            for (const auto& [key, shader] : cached_shaders) {
                const auto [k, v] = key;
                if (v == ShaderMark::Mesh)
                    requireTriangObj = true;
                if (v == ShaderMark::Sphere)
                    requireSphereObj = true;
                if (v == ShaderMark::Volume)
                    requireVolumeObj = true;
            }

            bool requireSphereLight = false;
            bool requireTriangLight = false;
            
            for (const auto& [_, ld] : xinxinoptix::get_lightdats()) {

                const auto shape_enum = magic_enum::enum_cast<zeno::LightShape>(ld.shape).value_or(zeno::LightShape::Point);

                if (shape_enum == zeno::LightShape::Sphere) {
                    requireSphereLight = true;
                } else if (shape_enum != zeno::LightShape::Point) {
                    requireTriangLight = true;
                }

                if (requireSphereLight && requireTriangLight) {
                    break;
                }
                continue;
            }

            if (requireTriangLight) {
                auto tmp = std::make_shared<ShaderPrepared>();

                tmp->filename = _light_shader_template.name;
                tmp->callable = _default_callable_template.shadtmpl;
                tmp->mark = ShaderMark::Mesh;
                tmp->matid = "Light";

                //cachedMaterial[ std::tuple{"Light", ShaderMark::Mesh} ] = tmp;
            }

            if (requireSphereLight) {
                auto tmp = std::make_shared<ShaderPrepared>();

                tmp->filename = _light_shader_template.name;
                tmp->callable = _default_callable_template.shadtmpl;
                tmp->mark = ShaderMark::Sphere;
                tmp->matid = "Light";
                
                //cachedMaterial[ std::tuple{"Light", ShaderMark::Sphere} ] = tmp;
            }

            std::vector<std::shared_ptr<ShaderPrepared>> allShaders{};
            allShaders.reserve(cached_shaders.size()+2);

            ShaderKeyIndex.clear();
            for (const auto& [key, shader] : cached_shaders) {
                auto idx = allShaders.size();

                allShaders.push_back(shader);
                ShaderKeyIndex[key] = idx;
                
                if (std::get<1>(key) == ShaderMark::Mesh)
                    meshMatLUT[std::get<0>(key)] = idx;
            }

            defaultScene.processVolumeBox(OptixUtil::context);
            defaultScene.shader_indice_table = ShaderKeyIndex;
            bool second_matNeedUpdate = zeno::getSession().userData().get2<bool>("viewport-optix-matNeedUpdate", true);
            if (matNeedUpdate && second_matNeedUpdate)
            {
                unsigned int usesPrimitiveTypeFlags = 0u;
                if (requireTriangObj || requireTriangLight)
                    usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
                if (requireSphereObj || requireSphereLight)
                    usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
                if (requireVolumeObj)
                    usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
                if (usesCurveTypeFlags)
                    usesPrimitiveTypeFlags |= usesCurveTypeFlags;

                auto refresh = OptixUtil::configPipeline((OptixPrimitiveTypeFlags)usesPrimitiveTypeFlags);
                ShaderDirty |= refresh;

                if (refresh) {
                    for (auto& shader : allShaders)
                        shader->dirty = true;
                }
                
                xinxinoptix::updateShaders(allShaders, 
                                                    requireTriangObj, requireTriangLight, 
                                                    requireSphereObj, requireSphereLight, 
                                                    requireVolumeObj, usesCurveTypeFlags, refresh);
                                                    
                defaultScene.prepareVolumeAssets();
            }

            bool bMeshMatLUTChanged = false;    //if meshMatLUT need update
            if (scene->drawOptions->updateMatlOnly) {
                bMeshMatLUTChanged = meshMatLUTChanged(meshMatLUT);
            }
            if (bMeshMatLUTChanged || matNeedUpdate && (staticNeedUpdate || meshNeedUpdate)) {
                std::map<std::string, uint16_t>().swap(cachedMeshMatLUT);
                cachedMeshMatLUT = meshMatLUT;
            }

            if (meshNeedUpdate || bMeshMatLUTChanged)
            {
                defaultScene.updateMeshMaterials(meshMatLUT);

                xinxinoptix::updateCurves();
                xinxinoptix::prepareScene();

                xinxinoptix::configPipeline(ShaderDirty);
                std::cout<< "Finish optix update" << std::endl;
            }

            if (scene->drawOptions->updateMatlOnly && !bMeshMatLUTChanged)
            {

                xinxinoptix::configPipeline(ShaderDirty);
                std::cout << "Finish optix update" << std::endl;
            }

        }

        if(lightNeedUpdate){
            CppTimer timer; timer.tick();
            xinxinoptix::buildLightTree();
//            timer.tock("Build LightTree");
        }

        if (lightNeedUpdate || matNeedUpdate || meshNeedUpdate || staticNeedUpdate) {

            lightNeedUpdate = false;
            xinxinoptix::updateRootIAS();

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
        xinxinoptix::optixDestroy();
    }

    void assetLoad() {
        int a = 0;
        defaultScene = {};
    }

    void run() {
        int a = 0;
    }

    void beginFrameLoading(int frameid) {
        int a = 0;
    }

    void endFrameLoading(int frameid) {
        int a = 0;
    }

    void cleanupAssets() override {
        cached_shaders = {};
        OptixUtil::rtMaterialShaders.clear();

        xinxinoptix::optixCleanup();
    }

    void cleanupWhenExit() override {

    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineOptx>("optx");

} // namespace zenovis::optx
#endif
