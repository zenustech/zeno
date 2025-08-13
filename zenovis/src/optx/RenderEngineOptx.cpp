#include <tuple>
#include <unordered_map>
#include <vcruntime_string.h>
#include <vector_types.h>
#ifdef ZENO_ENABLE_OPTIX

#include "Scene.h"
#include <tsl/ordered_map.h>
#include "xinxinoptixapi.h"
#include "zeno/utils/vec.h"
#include <limits>
#include <memory>
#include <tbb/mutex.h>
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

#include <curve/Hair.h>
#include <curve/optixCurve.h>

#include "ShaderBuffer.h"
#include <zeno/extra/ShaderNode.h>
static bool recordedSimpleRender = false;
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
float norm_infvec2(zeno::vec3f &p1,  zeno::vec3f &p2)
{
    return std::max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]) );
}
static CppTimer timer, localTimer;
static void cleanMesh(zeno::PrimitiveObject* prim,
               std::vector<zeno::vec3f> &verts,
               std::vector<zeno::vec3f> &nrm,
               std::vector<zeno::vec3f> &clr,
               std::vector<zeno::vec3f> &tang,
               std::vector<zeno::vec3f> &uv,
               std::vector<zeno::vec3i> &idxBuffer)
{
    const bool has_clr = prim->has_attr("clr");
    float tol = 1e-5;
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
          if(norm_infvec2(tester, uv)<tol )
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
  verts.reserve(count);
  nrm.reserve(count);
    if (has_clr) {
        clr.reserve(count);
    }
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
      auto t  = prim->verts.attr<zeno::vec3f>("atang")[vid];
      verts.push_back(v);
      nrm.push_back(n);
        if (has_clr) {
            auto c  = prim->verts.attr<zeno::vec3f>("clr")[vid];
            clr.push_back(c);
        }
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
          if(norm_infvec2(tuv, vuv)<tol)
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
                    const auto reName = prim_in->userData().get2<std::string>("ObjectName", abcpath);

                    defaultScene.preloadCurveGroup(points, widths, normals, strands, curveTypeEnum, reName);
                    return;
                }

                auto is_cyhair = prim_in_lslislSp->userData().has("cyhair");
                if (is_cyhair) {
                    auto& ud = prim_in_lslislSp->userData();
                    const auto objectName = prim_in->userData().get2<std::string>("ObjectName", key); 

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
                    defaultScene.preloadHair( objectName, path_string, type_index, trans);
                    return;
                }

                auto is_sphere = prim_in_lslislSp->userData().has("sphere_center");
                if (is_sphere) {

                    auto& ud = prim_in_lslislSp->userData();

                    auto sphere_scale = ud.get2<zeno::vec3f>("sphere_scale");
                    auto uniform_scaling = sphere_scale[0] == sphere_scale[1] && sphere_scale[2] == sphere_scale[0];
                    {
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

                        defaultScene.preload_sphere(objectName, sphere_transform);
                    }
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
                    auto count = prim_in->verts->size() / 4;

                    std::vector<m3r4c> matrix_list(count);

                    std::copy_n((float*)prim_in->verts.data(), count * 12, (float*)matrix_list.data());

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
                        has_uv = true;
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
                    if (!has_uv) {
                        prim_in->add_attr<zeno::vec3f>("uv");
                    }
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
                    if (!clr.empty()) {
                        oPrim->add_attr<zeno::vec3f>("clr");
                        oPrim->verts.attr<zeno::vec3f>("clr") = clr;
                    }
                    oPrim->tris.resize(idxBuffer.size());
                    oPrim->verts.attr<zeno::vec3f>("pos") = verts;
                    oPrim->verts.attr<zeno::vec3f>("nrm") = nrm;
                    oPrim->tris = idxBuffer;
                    
                    if (has_uv) {
                        oPrim->add_attr<zeno::vec3f>("uv");
                        oPrim->add_attr<zeno::vec3f>("atang");
                        oPrim->verts.attr<zeno::vec3f>("uv") = uv;
                        oPrim->verts.attr<zeno::vec3f>("atang") = tang;
                    }
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
                            auto matName = prim_in->userData().get2<std::string>(matIdx, "");
                            matNameList.emplace_back(matName);
                        }
                    }
                    auto mtlid = prim_in->userData().get2<std::string>("mtlid", "");
                    if ("" == mtlid) {
                        mtlid = prim_in->userData().get2<std::string>("Material_0", "");
                    }
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
                
                DetMaterial detm {}; 
                detm.tex2Ds = mtl->tex2Ds; 
                detm.tex3Ds = mtl->tex3Ds; 
                detm.common = mtl->common; 
                detm.shader = mtl->frag;
                detm.extensions = mtl->extensions;
                detm.mtlidkey = mtl->mtlidkey;
                detm.parameters = mtl->parameters;
                detm.stamp_base = stamp_base;
                detm.dirty = dirty;
                
                det = std::move(detm);
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
                    OptixUtil::sky_tex = path;
                } else {
                    OptixUtil::sky_tex = OptixUtil::default_sky_tex;
                }
                OptixUtil::setSkyTexture(OptixUtil::sky_tex.value());

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
                            zeno::getSession().userData().has("optix_image_path")?1:ud.get2<int>("renderRatio"),
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

        timer.tick();
        //tbb::task_arena limited(12);
        tbb::task_group run_group;
        tbb::mutex ig_mutex;
        
        for (auto const &[key, obj] : objs) {

            run_group.run([&]() {
                    
                ig_mutex.lock();
                auto may = ins.may_emplace(key);
                ig_mutex.unlock();

                if (may && key.find(":static:") == key.npos) {
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
                                zeno::getSession().userData().has("optix_image_path")?1:ud.get2<int>("renderRatio"),
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
                    ig_mutex.lock();
                    ins.try_emplace(key, std::move(ig));
                    ig_mutex.unlock();
                }
            });
        }
        run_group.wait();
        timer.tock("Objects load");

        {   //when turn off last node in always mode
            static int objsNum = 0;
            if (objsNum > objs.size() && !changed)
                changed = true;
            objsNum = objs.size();
        }
        // return ins.has_changed();
        return changed;
    }
    void load_matrix_objects(std::vector<std::shared_ptr<zeno::IObject>> matrixs) {
        std::unordered_map<std::string, std::shared_ptr<zeno::IObject>> map;
        for (auto i = 0; i < matrixs.size(); i++) {
            if (matrixs[i]->userData().get2<std::string>("ResourceType", "") != "Matrixes") {
                continue;
            }
            auto obj_name = matrixs[i]->userData().get2<std::string>("ObjectName", "");
            if (obj_name == "") {
                continue;
            }
            if (auto mat = std::dynamic_pointer_cast<zeno::PrimitiveObject>(matrixs[i])) {
                auto count = mat->verts->size() / 4;
                std::vector<m3r4c> matrix_list(count);
                std::copy_n((float*)mat->verts.data(), count * 12, (float*)matrix_list.data());
                defaultScene.load_matrix_list(obj_name, matrix_list);
            }
            map[obj_name] = matrixs[i];
        }
        for (auto &[k, v]: graphics.m_curr) {
            if (auto* ptr = std::get_if<DetPrimitive>(&v->det)) {
                if (ptr == nullptr) {
                    continue;
                }
                if (ptr->primSp == nullptr) {
                    continue;
                }
                auto obj_name = ptr->primSp->userData().get2<std::string>("ObjectName", "");
                if (map.count(obj_name)) {
                    auto prim_ptr = std::dynamic_pointer_cast<zeno::PrimitiveObject>(map[obj_name]);
                    if (ptr->primSp->verts.size() == prim_ptr->verts.size()) {
                        ptr->primSp->verts = prim_ptr->verts;
                    }
                }
            }
        }

    }
};

static std::optional<glm::vec3> hitOnPlane(glm::vec3 ori, glm::vec3 dir, glm::vec3 n, glm::vec3 p) {
    auto t = glm::dot((p - ori), n) / glm::dot(dir, n);
    if (t > 0)
        return ori + dir * t;
    else
        return {};
}

// x, y from [0, 1]
static glm::vec3 screenPosToRayWS(float x, float y, float fov_degree, glm::vec2 res, glm::quat rot)  {
    x = (x - 0.5) * 2;
    y = (y - 0.5) * (-2);
    float v = std::tan(glm::radians(fov_degree) * 0.5f);
    float aspect = res.x / res.y;
    auto dir = glm::normalize(glm::vec3(v * x * aspect, v * y, -1));
    return rot * dir;
}

static std::optional<glm::vec3> get_proj_pos_on_plane(
        Json const &in_msg
        , float pos_x
        , float pos_y
        , zenovis::Camera *camera
        , glm::vec3 const & pivot
        , glm::vec3 const & plane_dir
) {
    float res_x = in_msg["Resolution"][0];
    float res_y = in_msg["Resolution"][1];
    auto ori = camera->getPos();
    auto fov_degree = camera->m_fov;
    auto rot = camera->m_rotation;
    glm::vec3 dir = screenPosToRayWS(
        pos_x / res_x
        , pos_y / res_y
        , fov_degree
        , {res_x, res_y}
        , rot
    );
    std::optional<glm::vec3> t = hitOnPlane(ori, dir, plane_dir, pivot);
    return t;
}
std::optional<glm::quat> rotate(glm::vec3 start_vec, glm::vec3 end_vec, glm::vec3 axis) {
    start_vec = glm::normalize(start_vec);
    end_vec = glm::normalize(end_vec);
    if (glm::length(start_vec - end_vec) < 0.0001) {
        return std::nullopt;
    }
    auto cross_vec = glm::cross(start_vec, end_vec);
    float direct = 1.0f;
    if (glm::dot(cross_vec, axis) < 0) {
        direct = -1.0f;
    }
    float angle = acos(glm::clamp(glm::dot(start_vec, end_vec), -1.0f, 1.0f));
    glm::quat q(glm::rotate(angle * direct, axis));
    return q;
}

static glm::vec2 pos_ws2ss(glm::vec3 pos_WS, glm::mat4 const &vp_mat, glm::vec2 resolution) {
    auto pivot_CS = vp_mat * glm::vec4(pos_WS, 1.0f);
    glm::vec2 pivot_SS = (pivot_CS / pivot_CS[3]);
    pivot_SS = pivot_SS * 0.5f + 0.5f;
    pivot_SS[1] = 1 - pivot_SS[1];
    pivot_SS = pivot_SS * resolution;
    return pivot_SS;
}
struct RenderEngineOptx : RenderEngine, zeno::disable_copy {
    std::unique_ptr<GraphicsManager> graphicsMan;
#ifdef OPTIX_BASE_GL
    std::unique_ptr<opengl::VAO> vao;
#endif
    Scene *scene;

    std::unordered_map<std::string, std::vector<OptixUtil::TexKey>> shader_tex_stat;

    bool lightNeedUpdate = true;
    bool meshNeedUpdate = true;
    bool matNeedUpdate = true;
    bool staticNeedUpdate = true;
    void outlineInit(Json const &in_msg) override {
//        zeno::log_error("MessageType: {}", in_msg.dump());
        if (in_msg["MessageType"] == "Init") {
            Json message;
            if (!defaultScene.static_scene_tree.is_null()) {
                Json scene_tree;
                scene_tree["root_name"] = defaultScene.static_scene_tree["root_name"];
                scene_tree["scene_tree"] = defaultScene.static_scene_tree["scene_tree"];
                message["StaticSceneTree"] = scene_tree;
            }
            if (!defaultScene.dynamic_scene_tree.is_null()) {
                Json scene_tree;
                scene_tree["node_key"] = defaultScene.dynamic_scene_tree["node_key"];
                scene_tree["root_name"] = defaultScene.dynamic_scene_tree["root_name"];
                scene_tree["scene_tree"] = defaultScene.dynamic_scene_tree["scene_tree"];
                message["DynamicSceneTree"] = scene_tree;
            }

            if (message.is_null()) {
                return;
            }

            message["MessageType"] = "SceneTree";
            fun(message.dump());
        }
        else if (in_msg["MessageType"] == "Select") {
            auto &link = in_msg["Content"];
            {
                std::string object_name = link.back();
                Json *json = nullptr;
                if (link[0] == "StaticScene") {
                    json = &defaultScene.static_scene_tree;
                }
                else {
                    json = &defaultScene.dynamic_scene_tree;
                }
                Json &scene_tree = json->operator[]("scene_tree");
                Json &node_to_matrix = json->operator[]("node_to_matrix");
                glm::mat4 p_matrix = glm::mat4(1);
                glm::mat4 l_matrix = glm::mat4(1);
                for (auto idx = 1; idx < link.size(); idx++) {
                    auto &node_name = link[idx];
                    p_matrix = p_matrix * l_matrix;
                    if (defaultScene.modified_xfroms.count(node_name)) {
                        l_matrix = defaultScene.modified_xfroms[node_name];
                        continue;
                    }
                    auto matrix_node_json = scene_tree[node_name];
                    if (matrix_node_json.is_null()) {
                        break;
                    }
                    std::string matrix_name = matrix_node_json["matrix"];
                    auto mat_json = node_to_matrix[matrix_name][0];
                    for (auto i = 0; i < 4; i++) {
                        for (auto j = 0; j < 3; j++) {
                            int index = i * 3 + j;
                            l_matrix[i][j] = float(mat_json[index]);
                        }
                    }
                }
                defaultScene.cur_node = {object_name, l_matrix, p_matrix};
                {
                    Json message;
                    message["MessageType"] = "SetGizmoAxis";
                    auto g_mat = p_matrix * l_matrix;
                    message["r0"] = {g_mat[0][0], g_mat[0][1] , g_mat[0][2]};
                    message["r1"] = {g_mat[1][0], g_mat[1][1] , g_mat[1][2]};
                    message["r2"] = {g_mat[2][0], g_mat[2][1] , g_mat[2][2]};
                    message["t"]  = {g_mat[3][0], g_mat[3][1] , g_mat[3][2]};
                    fun(message.dump());
                }
            }
        }
        else if (in_msg["MessageType"] == "ResetNodeModify") {
            auto node_name = std::string(in_msg["NodeName"]);
            defaultScene.modified_xfroms.erase(node_name);
            std::string mat_name = node_name + "_m";
            std::vector<glm::mat4> matrixs = {glm::mat4(1)};
            if (defaultScene.dynamic_scene->node_to_matrix.count(mat_name)) {
                matrixs = defaultScene.dynamic_scene->node_to_matrix[mat_name];
                auto prim = defaultScene.dynamic_scene->mats_to_prim(mat_name, matrixs, false, "TotalChange");
                load_matrix_objects({prim});
            }
            if (defaultScene.cur_node.has_value()) {
                auto &[name, lmat, pmat] = defaultScene.cur_node.value();
                if (name == node_name) {
                    lmat = matrixs[0];
                    {
                        auto &[_name, _lmat, pmat] = defaultScene.cur_node.value();
                        Json message;
                        message["MessageType"] = "SetGizmoAxis";
                        auto g_mat = pmat * _lmat;
                        message["r0"] = {g_mat[0][0], g_mat[0][1] , g_mat[0][2]};
                        message["r1"] = {g_mat[1][0], g_mat[1][1] , g_mat[1][2]};
                        message["r2"] = {g_mat[2][0], g_mat[2][1] , g_mat[2][2]};
                        message["t"]  = {g_mat[3][0], g_mat[3][1] , g_mat[3][2]};
                        fun(message.dump());
                    }
                }
            }
        }
        else if (in_msg["MessageType"] == "Xform") {

//            zeno::log_info("Axis: {}", in_msg["Axis"]);
            if (!defaultScene.cur_node.has_value()) {
                return;
            }
            auto &[name, lmat, pmat] = defaultScene.cur_node.value();
            if (defaultScene.modified_xfroms.count(name) == 0) {
                defaultScene.modified_xfroms[name] = lmat;
            }
            auto g_mat = pmat * defaultScene.modified_xfroms[name];
            auto pivot = glm::vec3(g_mat * glm::vec4(0, 0, 0, 1));

            const auto x_axis = glm::vec3(1, 0, 0);
            const auto y_axis = glm::vec3(0, 1, 0);
            const auto z_axis = glm::vec3(0, 0, 1);

            std::string mode = in_msg["Mode"];
            bool is_local_space = in_msg["LocalSpace"];
            std::map<std::string, glm::vec3> axis_mapping = {
                {"X", {1, 0, 0}},
                {"Y", {0, 1, 0}},
                {"Z", {0, 0, 1}},
                {"XY", {1, 1, 0}},
                {"YZ", {0, 1, 1}},
                {"XZ", {1, 0, 1}},
                {"", {1, 1, 1}},
                {"XYZ", {1, 1, 1}},
            };

            glm::mat3 local_mat;
            local_mat[0] = glm::normalize(glm::vec3(g_mat[0]));
            local_mat[1] = glm::normalize(glm::vec3(g_mat[1]));
            local_mat[2] = glm::normalize(glm::vec3(g_mat[2]));

            std::optional<std::pair<std::string, glm::mat4>> result;
            if (mode == "RotateScreen" || (mode == "Rotate" && in_msg["Axis"]!="X" && in_msg["Axis"]!="Y" && in_msg["Axis"]!="Z") ) {
                glm::vec3 cam_pos = scene->camera->getPos();
                glm::vec3 cam_up = scene->camera->get_lodup();
                auto local_z = glm::normalize(cam_pos - pivot);
                auto local_x = glm::normalize(glm::cross(cam_up, local_z));
                auto local_y = glm::normalize(glm::cross(local_z, local_x));
                auto local_mat = glm::mat3(1);
                local_mat[0] = local_x;
                local_mat[1] = local_y;
                local_mat[2] = local_z;

                {
                    auto delta_x = float(in_msg["Delta"][0]);
                    auto delta_y = float(in_msg["Delta"][1]);
                    glm::mat4 xform = glm::rotate(glm::mat4(1.0f), glm::radians(delta_x), local_y);
                    xform = glm::rotate(xform, glm::radians(delta_y), local_x);
                    auto trans2local = glm::translate(glm::mat4(1), glm::vec3(-pivot));
                    auto trans2local_inv = glm::inverse(trans2local);
                    g_mat = trans2local_inv * xform * trans2local * g_mat;
                    auto n_mat = glm::inverse(pmat) * g_mat;
                    result = {name, n_mat};
                }
            }
            else if (is_local_space && !(mode == "Translate" && (in_msg["Axis"] == "" || in_msg["Axis"] == "XYZ"))) {
//                zeno::log_info("Axis: {}", in_msg["Axis"]);
                if (mode == "Translate") {
                    std::map<std::string, glm::vec3> selected_plane_dir_mapping = {
                        {"X", {0, 0, 1}},
                        {"Y", {0, 0, 1}},
                        {"Z", {0, 1, 0}},
                        {"XY", {0, 0, 1}},
                        {"YZ", {1, 0, 0}},
                        {"XZ", {0, 1, 0}},
                    };
                    glm::vec3 selected_plane_dir = scene->camera->get_lodfront();
                    if (selected_plane_dir_mapping.count(in_msg["Axis"])) {
                        selected_plane_dir = selected_plane_dir_mapping[in_msg["Axis"]];
                    }
                    selected_plane_dir = local_mat * selected_plane_dir;
                    auto trans_start = get_proj_pos_on_plane(
                        in_msg
                        , float(in_msg["LastPos"][0])
                        , float(in_msg["LastPos"][1])
                        , scene->camera.get()
                        , pivot
                        , selected_plane_dir
                    );
                    if (!trans_start.has_value()) {
                        return;
                    }
                    auto trans_end = get_proj_pos_on_plane(
                        in_msg
                        , float(in_msg["CurPos"][0])
                        , float(in_msg["CurPos"][1])
                        , scene->camera.get()
                        , pivot
                        , selected_plane_dir
                    );
                    if (!trans_end.has_value()) {
                        return;
                    }
                    auto trans = trans_end.value() - trans_start.value();
                    glm::vec3 axis = axis_mapping.at(in_msg["Axis"]);
                    auto proj_trans = glm::vec3();
                    for (auto i = 0; i < 3; i++) {
                        if (axis[i]) {
                            auto temp_axis = glm::vec3();
                            temp_axis[i] = 1;
                            temp_axis = local_mat * temp_axis;
                            proj_trans += glm::dot(trans, temp_axis) * temp_axis;
                        }
                    }
                    glm::mat4 xform = glm::mat4(1);
                    xform = glm::translate(glm::mat4(1), proj_trans);
                    auto trans2local = glm::translate(glm::mat4(1), glm::vec3(-pivot));
                    auto trans2local_inv = glm::inverse(trans2local);
                    g_mat = trans2local_inv * xform * trans2local * g_mat;
                    auto n_mat = glm::inverse(pmat) * g_mat;
                    result = {name, n_mat};
                }
                else if (mode == "Scale") {
                    auto vp = scene->camera->get_proj_matrix() * scene->camera->get_view_matrix();
                    auto resolution = glm::vec2(float(in_msg["Resolution"][0]), float(in_msg["Resolution"][1]));
                    auto pivot_SS = pos_ws2ss(pivot, vp, resolution);
                    glm::vec2 start_pos_SS = {float(in_msg["LastPos"][0]), float(in_msg["LastPos"][1])};
                    glm::vec2 end_pos_SS   = {float(in_msg["CurPos"][0]), float(in_msg["CurPos"][1])};
                    glm::vec3 axis = axis_mapping.at(in_msg["Axis"]);
                    if(in_msg["Axis"]==""||in_msg["Axis"]=="XYZ")
                        axis = {1,1,1};
                    auto start_len = glm::distance(pivot_SS, start_pos_SS);
                    if (start_len < 1) {
                        start_len = 1;
                    }
                    auto scale_size = glm::distance(pivot_SS, end_pos_SS) / start_len;
                    glm::vec3 scale(1.0f);
                    for (int i = 0; i < 3; i++) {
                        if (axis[i] == 1) {
                            scale[i] = std::max(scale_size, 0.1f);
                        }
                    }
                    glm::mat4 xform = glm::mat4(1);
                    xform = glm::scale(glm::mat4(1), scale);
                    auto n_mat = defaultScene.modified_xfroms[name] * xform;
                    result = {name, n_mat};
                }
                else if (mode == "Rotate") {
                    std::map<std::string, glm::vec3> selected_plane_dir_mapping = {
                        {"X", glm::vec3(1, 0, 0)},
                        {"Y", glm::vec3(0, 1, 0)},
                        {"Z", glm::vec3(0, 0, 1)},
                    };
                    std::string axis = in_msg["Axis"];
                    if (axis == "X" || axis == "Y" || axis == "Z") {
                        auto plane_dir = local_mat * selected_plane_dir_mapping[axis];
                        auto rot_start = get_proj_pos_on_plane(
                                in_msg, float(in_msg["LastPos"][0]), float(in_msg["LastPos"][1]), scene->camera.get(),
                                pivot, plane_dir
                        );
                        if (!rot_start.has_value()) {
                            return;
                        }
                        auto rot_end = get_proj_pos_on_plane(
                                in_msg, float(in_msg["CurPos"][0]), float(in_msg["CurPos"][1]), scene->camera.get(),
                                pivot, plane_dir
                        );
                        if (!rot_end.has_value()) {
                            return;
                        }
                        auto start_vec = rot_start.value() - pivot;
                        auto end_vec = rot_end.value() - pivot;
                        auto rot_quat = rotate(start_vec, end_vec, plane_dir);
                        if (!rot_quat.has_value()) {
                            return;
                        }
                        glm::mat4 xform = glm::toMat4(rot_quat.value());
                        auto trans2local = glm::translate(glm::mat4(1), glm::vec3(-pivot));
                        auto trans2local_inv = glm::inverse(trans2local);
                        g_mat = trans2local_inv * xform * trans2local * g_mat;
                        auto n_mat = glm::inverse(pmat) * g_mat;
                        result = {name, n_mat};
                    }
                }
            }
            else {
                if (mode == "Translate") {
                    std::map<std::string, glm::vec3> selected_plane_dir_mapping = {
                        {"X", {0, 0, 1}},
                        {"Y", {0, 0, 1}},
                        {"Z", {0, 1, 0}},
                        {"XY", {0, 0, 1}},
                        {"YZ", {1, 0, 0}},
                        {"XZ", {0, 1, 0}},
                    };
                    glm::vec3 selected_plane_dir = scene->camera->get_lodfront();
                    if (selected_plane_dir_mapping.count(in_msg["Axis"])) {
                        selected_plane_dir = selected_plane_dir_mapping[in_msg["Axis"]];
                    }
                    auto trans_start = get_proj_pos_on_plane(
                        in_msg
                        , float(in_msg["LastPos"][0])
                        , float(in_msg["LastPos"][1])
                        , scene->camera.get()
                        , pivot
                        , selected_plane_dir
                    );
                    if (!trans_start.has_value()) {
                        return;
                    }
                    auto trans_end = get_proj_pos_on_plane(
                        in_msg
                        , float(in_msg["CurPos"][0])
                        , float(in_msg["CurPos"][1])
                        , scene->camera.get()
                        , pivot
                        , selected_plane_dir
                    );
                    if (!trans_end.has_value()) {
                        return;
                    }
                    auto trans = trans_end.value() - trans_start.value();
                    glm::vec3 axis = axis_mapping.at(in_msg["Axis"]);
                    trans *= axis;
                    glm::mat4 xform = glm::mat4(1);
                    xform = glm::translate(glm::mat4(1), trans);
                    auto trans2local = glm::translate(glm::mat4(1), glm::vec3(-pivot));
                    auto trans2local_inv = glm::inverse(trans2local);
                    g_mat = trans2local_inv * xform * trans2local * g_mat;
                    auto n_mat = glm::inverse(pmat) * g_mat;
                    result = {name, n_mat};
                }
                else if (mode == "Scale") {
                    auto vp = scene->camera->get_proj_matrix() * scene->camera->get_view_matrix();
                    auto resolution = glm::vec2(float(in_msg["Resolution"][0]), float(in_msg["Resolution"][1]));
                    auto pivot_SS = pos_ws2ss(pivot, vp, resolution);
                    glm::vec2 start_pos_SS = {float(in_msg["LastPos"][0]), float(in_msg["LastPos"][1])};
                    glm::vec2 end_pos_SS   = {float(in_msg["CurPos"][0]), float(in_msg["CurPos"][1])};
                    glm::vec3 axis = axis_mapping.at(in_msg["Axis"]);
                    auto start_len = glm::distance(pivot_SS, start_pos_SS);
                    if (start_len < 1) {
                        start_len = 1;
                    }
                    auto scale_size = glm::distance(pivot_SS, end_pos_SS) / start_len;
                    glm::vec3 scale(1.0f);
                    for (int i = 0; i < 3; i++) {
                        if (axis[i] == 1) {
                            scale[i] = std::max(scale_size, 0.1f);
                        }
                    }
                    glm::mat4 xform = glm::mat4(1);
                    xform = glm::scale(glm::mat4(1), scale);
                    auto trans2local = glm::translate(glm::mat4(1), glm::vec3(-pivot));
                    auto trans2local_inv = glm::inverse(trans2local);
                    g_mat = trans2local_inv * xform * trans2local * g_mat;
                    auto n_mat = glm::inverse(pmat) * g_mat;
                    result = {name, n_mat};
                }
                else if (mode == "Rotate") {
                    std::map<std::string, glm::vec3> selected_plane_dir_mapping = {
                        {"X", {1, 0, 0}},
                        {"Y", {0, 1, 0}},
                        {"Z", {0, 0, 1}},
                    };
                    std::string axis = in_msg["Axis"];
                    if (axis == "X" || axis == "Y" || axis == "Z") {
                        auto plane_dir = selected_plane_dir_mapping[axis];
                        auto rot_start = get_proj_pos_on_plane(
                            in_msg
                            , float(in_msg["LastPos"][0])
                            , float(in_msg["LastPos"][1])
                            , scene->camera.get()
                            , pivot
                            , plane_dir
                        );
                        if (!rot_start.has_value()) {
                            return;
                        }
                        auto rot_end = get_proj_pos_on_plane(
                            in_msg
                            , float(in_msg["CurPos"][0])
                            , float(in_msg["CurPos"][1])
                            , scene->camera.get()
                            , pivot
                            , plane_dir
                        );
                        if (!rot_end.has_value()) {
                            return;
                        }
                        auto start_vec = rot_start.value() - pivot;
                        auto end_vec = rot_end.value() - pivot;
                        auto rot_quat = rotate(start_vec, end_vec, plane_dir);
                        if (!rot_quat.has_value()) {
                            return;
                        }
                        glm::mat4 xform = glm::toMat4(rot_quat.value());
                        auto trans2local = glm::translate(glm::mat4(1), glm::vec3(-pivot));
                        auto trans2local_inv = glm::inverse(trans2local);
                        g_mat = trans2local_inv * xform * trans2local * g_mat;
                        auto n_mat = glm::inverse(pmat) * g_mat;
                        result = {name, n_mat};
                    }
                }
            }

            if (result.has_value()) {
                std::string name = result.value().first;
                glm::mat4 n_mat = result.value().second;
                defaultScene.modified_xfroms[name] = n_mat;
                auto mat_prim = std::make_shared<zeno::PrimitiveObject>();
                mat_prim->verts.resize(4);
                mat_prim->verts[0][0] = n_mat[0][0];
                mat_prim->verts[0][1] = n_mat[1][0];
                mat_prim->verts[0][2] = n_mat[2][0];
                mat_prim->verts[1][0] = n_mat[3][0];
                mat_prim->verts[1][1] = n_mat[0][1];
                mat_prim->verts[1][2] = n_mat[1][1];
                mat_prim->verts[2][0] = n_mat[2][1];
                mat_prim->verts[2][1] = n_mat[3][1];
                mat_prim->verts[2][2] = n_mat[0][2];
                mat_prim->verts[3][0] = n_mat[1][2];
                mat_prim->verts[3][1] = n_mat[2][2];
                mat_prim->verts[3][2] = n_mat[3][2];

                mat_prim->userData().set2("ResourceType", std::string("Matrixes"));
                mat_prim->userData().set2("ObjectName", name+"_m");
                load_matrix_objects({mat_prim});
                {
                    Json xform_json;
                    xform_json["MessageType"] = "SetNodeXform";
                    xform_json["Mode"] = "Set";
                    xform_json["NodeKey"] = defaultScene.dynamic_scene_tree["node_key"];
                    xform_json["NodeName"] = name;
                    xform_json["r0"] = {n_mat[0][0], n_mat[0][1] , n_mat[0][2]};
                    xform_json["r1"] = {n_mat[1][0], n_mat[1][1] , n_mat[1][2]};
                    xform_json["r2"] = {n_mat[2][0], n_mat[2][1] , n_mat[2][2]};
                    xform_json["t"]  = {n_mat[3][0], n_mat[3][1] , n_mat[3][2]};
                    fun(xform_json.dump());
                }
                {
                    auto &[_name, _lmat, pmat] = defaultScene.cur_node.value();
                    auto l_matrix = result.value().second;
                    Json message;
                    message["MessageType"] = "SetGizmoAxis";
                    auto g_mat = pmat * l_matrix;
                    message["r0"] = {g_mat[0][0], g_mat[0][1] , g_mat[0][2]};
                    message["r1"] = {g_mat[1][0], g_mat[1][1] , g_mat[1][2]};
                    message["r2"] = {g_mat[2][0], g_mat[2][1] , g_mat[2][2]};
                    message["t"]  = {g_mat[3][0], g_mat[3][1] , g_mat[3][2]};
                    fun(message.dump());
                }
            }
        }
        else if (in_msg["MessageType"] == "NeedSetSceneXform") {
            Json message;
            message["MessageType"] = "SetSceneXform";
            message["NodeKey"] = defaultScene.dynamic_scene_tree["node_key"];
            Json matrixs;
            for (const auto &[id, mat]: defaultScene.modified_xfroms) {
                Json matrix;
                for (auto i = 0; i < 4; i++) {
                    for (auto j = 0; j < 3; j++) {
                        matrix.push_back(mat[i][j]);
                    }
                }
                matrixs[id] = matrix;
            }
            message["Matrixs"] = matrixs;
            if (defaultScene.modified_xfroms.size()) {
				fun(message.dump());
            }
        }
        else if (in_msg["MessageType"] == "XformPanelInit") {
            Json message;
            message["MessageType"] = "XformPanelInitFeedback";
            Json matrixs;
            for (const auto &[id, mat]: defaultScene.modified_xfroms) {
                Json matrix;
                for (auto i = 0; i < 4; i++) {
                    for (auto j = 0; j < 3; j++) {
                        matrix.push_back(mat[i][j]);
                    }
                }
                matrixs[id] = Json::array();
                matrixs[id].push_back(matrix);
            }
            message["Matrixs"] = matrixs;
            fun(message.dump());
        }
    }
    std::optional<glm::vec3> getClickedPos(float x, float y) override {
        glm::vec3 posWS = xinxinoptix::get_click_pos(x, y);
        if (posWS == glm::vec3()) {
            return {};
        }
        auto const &cam = *scene->camera;
        posWS += cam.m_pos;
        return posWS;
    }
    std::optional<std::tuple<std::string, uint32_t, uint32_t>> getClickedId(float x, float y) override {
        auto ids = xinxinoptix::get_click_id(x, y);
        if (ids == glm::uvec4()) {
            return {};
        }
        uint64_t obj_id = *reinterpret_cast<uint64_t *>(&ids);
        if (defaultScene.gas_to_obj_id.count(obj_id)) {
            auto name = defaultScene.gas_to_obj_id.at(obj_id);
            return std::tuple<std::string, uint32_t, uint32_t>(name, ids[2], ids[3]);
        }
        return {};
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
    void load_matrix_objects(std::vector<std::shared_ptr<zeno::IObject>> matrixs) override {
        if (matrixs.empty()) {
            return;
        }
        graphicsMan->load_matrix_objects(matrixs);
        meshNeedUpdate = true;
    };

    void update_json(std::vector<std::pair<std::string, zeno::IObject *>> const &objs) {
        for (auto const&[key, obj]: objs) {
            Json message;
            message["MessageType"] = "SceneTree";
            if (obj == nullptr) {
                continue;
            }
            auto &ud = obj->userData();
            if (ud.get2<std::string>("ResourceType", "") == "SceneTree") {
                if (ud.get2<std::string>("SceneTreeType", "") == "static") {
                    if (!defaultScene.static_scene_tree.is_null()) {
                        continue;
                    }
                    auto content = ud.get2<std::string>("json");
                    defaultScene.static_scene_tree = Json::parse(content);
                    Json scene_tree;
                    scene_tree["root_name"] = defaultScene.static_scene_tree["root_name"];
                    scene_tree["scene_tree"] = defaultScene.static_scene_tree["scene_tree"];
                    message["StaticSceneTree"] = scene_tree;
                }
                else if (ud.get2<std::string>("SceneTreeType", "") == "dynamic") {
                    auto content = ud.get2<std::string>("json");
                    defaultScene.dynamic_scene_tree = Json::parse(content);
                    defaultScene.dynamic_scene_tree["node_key"] = key;

                    Json scene_tree;
                    scene_tree["root_name"] = defaultScene.dynamic_scene_tree["root_name"];
                    scene_tree["scene_tree"] = defaultScene.dynamic_scene_tree["scene_tree"];
                    scene_tree["node_key"] = defaultScene.dynamic_scene_tree["node_key"];
                    message["DynamicSceneTree"] = scene_tree;

                    defaultScene.dynamic_scene->from_json(defaultScene.dynamic_scene_tree);
                }
                else {
                    continue;
                }
                auto msg_str = message.dump();
                fun(std::move(msg_str));
            }
        }
        {
			Json message;
			message["MessageType"] = "XformPanelInitFeedback";
			message["Matrixs"] = Json::object();
			fun(message.dump());
        }
    }


    void replace_with_modified_matrix() {
        if (defaultScene.modified_xfroms.empty()) {
            return;
        }
        std::vector<std::shared_ptr<zeno::IObject>> mat_prims;
        for (auto const&[name, n_mat]: defaultScene.modified_xfroms) {
            auto mat_prim = std::make_shared<zeno::PrimitiveObject>();
            mat_prim->verts.resize(4);
            mat_prim->verts[0][0] = n_mat[0][0];
            mat_prim->verts[0][1] = n_mat[1][0];
            mat_prim->verts[0][2] = n_mat[2][0];
            mat_prim->verts[1][0] = n_mat[3][0];
            mat_prim->verts[1][1] = n_mat[0][1];
            mat_prim->verts[1][2] = n_mat[1][1];
            mat_prim->verts[2][0] = n_mat[2][1];
            mat_prim->verts[2][1] = n_mat[3][1];
            mat_prim->verts[2][2] = n_mat[0][2];
            mat_prim->verts[3][0] = n_mat[1][2];
            mat_prim->verts[3][1] = n_mat[2][2];
            mat_prim->verts[3][2] = n_mat[3][2];

            mat_prim->userData().set2("ResourceType", std::string("Matrixes"));
            mat_prim->userData().set2("ObjectName", name+"_m");
            mat_prims.push_back(mat_prim);
        }
        load_matrix_objects(mat_prims);
    }


    void update() override {
//        zeno::log_error("update");
        update_json(scene->objectsMan->pairs());

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
        }
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
        graphicsMan->load_shader_uniforms(scene->objectsMan->pairs());
        replace_with_modified_matrix();
    }

#define MY_CAM_ID(cam) cam.m_nx, cam.m_ny, cam.zOptixCameraSettingInfo.renderRatio, cam.m_rotation, cam.m_pos, cam.m_fov, cam.focalPlaneDistance, cam.m_aperture
#define MY_SIZE_ID(cam) cam.m_nx, cam.m_ny,cam.zOptixCameraSettingInfo.renderRatio
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

    std::map<std::string, std::set<ShaderMark>> required_shader_names;
    tsl::ordered_map<shader_key_t, std::shared_ptr<ShaderPrepared>, ByShaderKey> cached_shaders{};

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
            if(scene->drawOptions->simpleRender!=recordedSimpleRender)
            {
                sizeNeedUpdate = true;
                recordedSimpleRender = scene->drawOptions->simpleRender;
            }
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
            auto scale = zeno::getSession().userData().has("optix_image_path")?1:cam.zOptixCameraSettingInfo.renderRatio;
            scale = scene->drawOptions->simpleRender?scale:1;
            xinxinoptix::set_window_size(max(cam.m_nx/scale,1), max(cam.m_ny/scale,1));
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
                cam.zOptixCameraSettingInfo.renderRatio,
                cam.zOptixCameraSettingInfo.aces,
                cam.zOptixCameraSettingInfo.exposure,
                cam.zOptixCameraSettingInfo.panorama_camera,
                cam.zOptixCameraSettingInfo.panorama_vr180,
                cam.zOptixCameraSettingInfo.pupillary_distance
            );
        }
        bool second_matNeedUpdate = zeno::getSession().userData().get2<bool>("viewport-optix-matNeedUpdate", false);
        second_matNeedUpdate = second_matNeedUpdate || cached_shaders.empty();
        if ((meshNeedUpdate || matNeedUpdate || staticNeedUpdate) && second_matNeedUpdate) {

            std::unordered_map<shader_key_t, uint16_t, ByShaderKey> ShaderKeyIndex{};

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
                default_shader->matid = "";
                default_shader->filename = _default_shader_template.name;
                default_shader->callable = _default_callable_template.shadtmpl;
                return default_shader;
            };

            bool ShaderDirty = false;
            std::vector<std::string> dirtyShaderNames {};
            const auto shaderCount = cached_shaders.size();
            
            bool requireTriangObj = false;
            bool requireSphereObj = false;
            bool requireVolumeObj = false;

            unsigned int usesCurveTypeFlags = 0;
            auto curve_task = [&usesCurveTypeFlags](zeno::CurveType ele) {
                usesCurveTypeFlags |= CURVE_FLAG_MAP.at(ele);
                return CURVE_SHADER_MARK.at(ele);
            };
            
            if ( matNeedUpdate ) {
                required_shader_names = defaultScene.prepareShaderSet();

                dirtyShaderNames.clear();
                dirtyShaderNames.reserve(matMap.size());
            
                for (const auto& [key, value] : required_shader_names) {

                    bool is_default = matMap.count(key) == 0;
                    bool is_dirty = false;

                    if (!is_default) {
                        auto& shader_ref = matMap[key]; 
                        is_dirty = shader_ref->dirty;
                        if (is_dirty) {
                            dirtyShaderNames.push_back(key);
                        }
                    }

                    for (const auto& mark : value) {

                        if (mark > ShaderMark::Volume)
                        {
                            auto zmark = mark - 3;
                            curve_task((zeno::CurveType)zmark);
                        }

                        if (mark == ShaderMark::Mesh)
                            requireTriangObj = true;
                        if (mark == ShaderMark::Sphere)
                            requireSphereObj = true;
                        if (mark == ShaderMark::Volume)
                            requireVolumeObj = true;

                        auto shader_key = std::tuple {key, mark};
                        
                        if (is_default) {
                            auto shader_ref = cached_shaders[shader_key];
                            if (shader_ref==nullptr) {
                                cached_shaders[shader_key] = make_default_shader(mark);
                                ShaderDirty = true;
                            }
                        } else {
                            if (!is_dirty) continue;                             
                            cached_shaders[shader_key] = nullptr;
                            ShaderDirty = true;
                        }
                    }
                }
 
            } // preserve material names for materials-only updating case

            bool requireSphereLight = false;
            bool requireTriangLight = false;
            
            {   timer.tick();

                std::unordered_set<OptixUtil::TexKey, OptixUtil::TexKeyHash> requiredTexPathSet;
                for(auto const &matkey : dirtyShaderNames) {
                    if (required_shader_names.count( matkey ) == 0) continue;

                    const auto& texs = matMap[matkey]->tex2Ds;
                    for(auto& tex: texs) {
                        requiredTexPathSet.insert( {tex->path, tex->blockCompression} );
                    }
                }
                
                for (const auto& [_, ld] : xinxinoptix::get_lightdats()) {

                    if (!ld.textureKey.empty()) {
                        requiredTexPathSet.insert( {ld.textureKey, false} );
                    }
                    if (requireSphereLight && requireTriangLight) continue;
                    const auto shape_enum = magic_enum::enum_cast<zeno::LightShape>(ld.shape).value_or(zeno::LightShape::Point);
    
                    if (shape_enum == zeno::LightShape::Sphere)
                        requireSphereLight = true;
                    else if (shape_enum != zeno::LightShape::Point)
                        requireTriangLight = true;
                }
                
                tbb::task_group texture_group;
                for (const auto& key: requiredTexPathSet) {
                    texture_group.run([&]() {
                        OptixUtil::addTexture(key.path, key.blockCompression);
                    });
                }
                texture_group.wait();
                timer.tock("Texture load");
            }

            for(auto const &shaderName : dirtyShaderNames)
            {   
                //if (matMap.count(shaderName) == 0) continue;
                auto mtldet  = matMap[shaderName];

                if ( !mtldet->dirty ) continue;
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
                        shaderP.texs.reserve(mtldet->tex2Ds.size());

                        for (auto tex : mtldet->tex2Ds)
                        {
                            auto find = OptixUtil::tex_lut.find({ tex->path, tex->blockCompression });

                            if (find != OptixUtil::tex_lut.end()) {
                                auto& tex_ptr = find->second;
                                shaderP.texs.push_back(tex_ptr);
                            }
                            else {
                                shaderP.texs.push_back(nullptr);
                            }
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

                        auto& reuiredSet = required_shader_names.at(mtldet->mtlidkey);
                        
                        for (auto& mark : reuiredSet) {
                            shaderP.mark = mark;
                            auto _shader_key = std::tuple{mtldet->mtlidkey, mark};                            
                            cached_shaders[_shader_key] = std::make_shared<ShaderPrepared>(shaderP);
                        }
                    }
            }

            const auto prepareLightShader = [&](ShaderMark smark) {
                const auto shader_key = std::tuple{ "Light", smark };
                if (cached_shaders.count(shader_key)>0) return;

                auto tmp = std::make_shared<ShaderPrepared>();

                tmp->filename = _light_shader_template.name;
                tmp->callable = _default_callable_template.shadtmpl;
                tmp->mark = smark;
                tmp->matid = "Light";
                
                cached_shaders[shader_key] = tmp;
                ShaderDirty = true;
            };

            if (requireTriangLight)
                prepareLightShader(ShaderMark::Mesh);
            if (requireSphereLight)
                prepareLightShader(ShaderMark::Sphere);

            ShaderDirty |= cached_shaders.size() != shaderCount;

            std::vector<std::shared_ptr<ShaderPrepared>> allShaders{};
            allShaders.reserve(cached_shaders.size()+2);

            ShaderKeyIndex.clear();
            for (const auto& [key, shader] : cached_shaders) {
                auto idx = allShaders.size();

                allShaders.push_back(shader);
                ShaderKeyIndex[key] = idx;
            }

            defaultScene.load_shader_indice_table(ShaderKeyIndex);

                if(lightNeedUpdate){
                    timer.tick();
                    xinxinoptix::buildLightTree();
                    timer.tock("Build LightTree");
                }

                if (OptixUtil::tex_lut.size()>1) {

                    timer.tick();
                    std::vector<OptixUtil::TexKey> dtexs;
                    for (auto& [k, ptr] : OptixUtil::tex_lut) {
                        if (ptr!=nullptr && ptr.use_count()<=1)
                            dtexs.push_back(k);
                    }
                    for (auto& k : dtexs) {
                        OptixUtil::removeTexture(k);
                    }
                    timer.tock("Texture unload");
                }

            if (matNeedUpdate)
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

                if (ShaderDirty) {
                    xinxinoptix::updateShaders(allShaders, 
                                                    requireTriangObj, requireTriangLight, 
                                                    requireSphereObj, requireSphereLight, 
                                                    requireVolumeObj, usesCurveTypeFlags, refresh);
                }                                    
                defaultScene.prepareVolumeAssets();
            }

            if (meshNeedUpdate)
            {
                defaultScene.updateMeshMaterials();
                xinxinoptix::prepareScene();
            }

            if (matNeedUpdate || scene->drawOptions->updateMatlOnly)
            {
                xinxinoptix::configPipeline(ShaderDirty);
                std::cout<< "Finish optix update" << std::endl;
            }
        }
            if (meshNeedUpdate)
            {
                defaultScene.updateMeshMaterials();
                xinxinoptix::prepareScene();
            }

            lightNeedUpdate = false;
           
            matNeedUpdate = false;
            meshNeedUpdate = false;
            staticNeedUpdate = false;

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

        defaultScene = {};
        cached_shaders = {};
        OptixUtil::rtMaterialShaders.clear();
        {
            Json message;
            message["MessageType"] = "CleanupAssets";
            fun(message.dump());
        }
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
        {
            Json message;
            message["MessageType"] = "CleanupAssets";
            fun(message.dump());
        }
    }

    void cleanupWhenExit() override {

    }
};

static auto definer = RenderManager::registerRenderEngine<RenderEngineOptx>("optx");

} // namespace zenovis::optx
#endif
