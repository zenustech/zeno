#include <utility>
#include <zeno/zeno.h>
#include <zeno/types/CameraObject.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/log.h>

#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/LightObject.h>

#include "assimp/scene.h"

#include "Definition.h"
#include "tinygltf/json.hpp"

#include <memory>

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <fstream>

#define SET_CAMERA_DATA                         \
    out_pos = (n->pos);                       \
    out_up = (n->up);                         \
    out_view = (n->view);                     \
    out_fov = (n->fov);                       \
    out_aperture = (n->aperture);             \
    out_focalPlaneDistance = (n->focalPlaneDistance); \

namespace zeno {
namespace {

struct CihouMayaCameraFov : INode {
    virtual void apply() override {
        auto m_fit_gate = array_index_safe({"Horizontal", "Vertical"},
                                           get_input2<std::string>("fit_gate"),
                                           "fit_gate") + 1;
        auto m_focL = get_input2<float>("focL");
        auto m_fw = get_input2<float>("fw");
        auto m_fh = get_input2<float>("fh");
        auto m_nx = get_input2<float>("nx");
        auto m_ny = get_input2<float>("ny");
        float c_fov = 0;
        float c_aspect = m_fw/m_fh;
        float u_aspect = m_ny&&m_nx? m_nx/m_ny : c_aspect;
        zeno::log_info("cam nx {} ny {} fw {} fh {} aspect {} {}",
                       m_nx, m_ny, m_fw, m_fh, u_aspect, c_aspect);
        std::cout << "m_fit_gate:" << m_fit_gate << "\n";
        std::cout << "u_aspect:" << u_aspect << "\n";
        std::cout << "c_aspect:" << c_aspect << "\n";
        if(m_fit_gate == 1){
            c_fov = 2.0f * std::atan(m_fh/(u_aspect/c_aspect) / (2.0f * m_focL) ) * (180.0f / M_PI);
        }else if(m_fit_gate == 2){
            c_fov = 2.0f * std::atan(m_fw/c_aspect / (2.0f * m_focL) ) * (180.0f / M_PI);
        }
        set_output2("fov", c_fov);
    }
};

ZENO_DEFNODE(CihouMayaCameraFov)({
    {
        {"enum Horizontal Vertical", "fit_gate", "Horizontal"},
        {"float", "focL", "35"},
        {"float", "fw", "36"},
        {"float", "fh", "24"},
        {"float", "nx", "0"},
        {"float", "ny", "0"},
    },
    {
        {"float", "fov"},
    },
    {},
    {"FBX"},
});

std::shared_ptr<zeno::CurveObject> createCurvePoint(std::vector<float> &t, std::vector<float> &y)
{
    std::vector<zeno::vec2f> c1;
    std::vector<zeno::vec2f> c2;
    size_t N = t.size();
    c1.resize(N);
    c2.resize(N);
    c1[0] = zeno::vec2f(t[0] + (t[1] - t[0])/3.0f, y[0]);
    c2[0] = zeno::vec2f(t[0], y[0]);
    c1[N-1] = zeno::vec2f(t[N-1], y[N-1]);
    c2[N-1] = zeno::vec2f(t[N-1] + (t[N-2] - t[N-1])/3.0f, y[N-1]);
    for(size_t i=1;i<=N-2;i++)
    {
        float extrap = (t[i] - t[i-1])>0?(t[i+1] - t[i] )/(t[i] - t[i-1]):0;
        zeno::vec2f R = zeno::vec2f(t[i], y[i]) + extrap * ( zeno::vec2f(t[i], y[i]) - zeno::vec2f(t[i-1], y[i-1]));
        zeno::vec2f T = 0.5f * (R + zeno::vec2f(t[i+1],y[i+1]));
        c1[i] = (0.45*zeno::vec2f(t[i], y[i]) + 0.55*T);
    }
    for(size_t i=1;i<=N-2;i++)
    {
        zeno::vec2f dir = zeno::vec2f(t[i], y[i]) - c1[i];
        float l = (t[i] - t[i-1])*0.55;
        float amp = abs(dir[0])>0.0f?l/abs(dir[0]):0.0f;
        c2[i] = zeno::vec2f(t[i], y[i]) + amp * dir;
    }
    auto curve = std::make_shared<CurveObject>();
    for (auto i = 0; i < N; i++) {
        curve->addPoint("x", t[i], y[i], zeno::CurveData::PointType::kBezier, c2[i] - zeno::vec2f(t[i],y[i]), c1[i] - zeno::vec2f(t[i],y[i]));
    }
    return curve;
}
struct CameraEval: zeno::INode {

    std::shared_ptr<zeno::CurveObject> curve_x = {};
    std::shared_ptr<zeno::CurveObject> curve_y = {};
    std::shared_ptr<zeno::CurveObject> curve_z = {};
    std::shared_ptr<zeno::CurveObject> curve_tx = {};
    std::shared_ptr<zeno::CurveObject> curve_ty = {};
    std::shared_ptr<zeno::CurveObject> curve_tz = {};
    std::shared_ptr<zeno::CurveObject> curve_vx = {};
    std::shared_ptr<zeno::CurveObject> curve_vy = {};
    std::shared_ptr<zeno::CurveObject> curve_vz = {};
    std::shared_ptr<zeno::CurveObject> curve_ux = {};
    std::shared_ptr<zeno::CurveObject> curve_uy = {};
    std::shared_ptr<zeno::CurveObject> curve_uz = {};
    std::shared_ptr<zeno::CurveObject> curve_fov = {};
    std::shared_ptr<zeno::CurveObject> curve_apertures = {};
    std::shared_ptr<zeno::CurveObject> curve_fPD = {};

    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = std::lround(get_input2<float>("frameid"));
        } else {
            frameid = getGlobalState()->frameid;
        }

        auto nodelist = get_input<zeno::ListObject>("nodelist")->get<zeno::CameraObject>();

        std::sort(nodelist.begin(), nodelist.end(), [](const auto &a, const auto &b)-> bool {
            auto a_frame = a->userData().get2<float>("frame");
            auto b_frame = b->userData().get2<float>("frame");
            return a_frame < b_frame;
        });

        int target_camera_count = 0;
        for (const auto &cam: nodelist) {
            target_camera_count += cam->userData().get2<int>("is_target", 0);
        }

        if (nodelist.size() == 1) {
            set_output("camera", nodelist[0]);
        }
        else if (frameid <= std::lround(nodelist[0]->userData().get2<float>("frame"))) {
             set_output("camera", nodelist[0]);
        }
        else if (frameid >= std::lround(nodelist.back()->userData().get2<float>("frame"))) {
            set_output("camera", nodelist.back());
        }
        else {
            if (curve_x == nullptr) {
                std::vector<float> ts(nodelist.size());
                std::vector<float> xs(nodelist.size());
                std::vector<float> ys(nodelist.size());
                std::vector<float> zs(nodelist.size());

                std::vector<float> xus(nodelist.size());
                std::vector<float> yus(nodelist.size());
                std::vector<float> zus(nodelist.size());

                std::vector<float> fovs(nodelist.size());
                std::vector<float> apertures(nodelist.size());
                std::vector<float> focalPlaneDistances(nodelist.size());

                std::vector<float> txs(nodelist.size());
                std::vector<float> tys(nodelist.size());
                std::vector<float> tzs(nodelist.size());
                if (nodelist.size() == target_camera_count) {
                    for(int i = 0; i < nodelist.size(); i++) {
                        auto const & cur_node = nodelist[i];
                        auto f = cur_node->userData().get2<float>("frame");
                        auto target = cur_node->userData().get2<zeno::vec3f>("target");
                        auto [x, y, z] = cur_node->pos;
                        auto [ux, uy, uz] = cur_node->up;
                        ts[i] = f;
                        xs[i] = x;
                        ys[i] = y;
                        zs[i] = z;
                        txs[i] = target[0];
                        tys[i] = target[1];
                        tzs[i] = target[2];
                        xus[i] = ux;
                        yus[i] = uy;
                        zus[i] = uz;
                        fovs[i] = cur_node->fov;
                        apertures[i] = cur_node->aperture;
                        focalPlaneDistances[i] = cur_node->focalPlaneDistance;
                    }
                    curve_x = createCurvePoint(ts, xs);
                    curve_y = createCurvePoint(ts, ys);
                    curve_z = createCurvePoint(ts, zs);
                    curve_tx = createCurvePoint(ts, txs);
                    curve_ty = createCurvePoint(ts, tys);
                    curve_tz = createCurvePoint(ts, tzs);
                    curve_ux = createCurvePoint(ts, xus);
                    curve_uy = createCurvePoint(ts, yus);
                    curve_uz = createCurvePoint(ts, zus);
                    curve_fov = createCurvePoint(ts, fovs);
                    curve_apertures = createCurvePoint(ts, apertures);
                    curve_fPD = createCurvePoint(ts, focalPlaneDistances);
                }
                else {
                    float totalLength = 0;
                    for(int i = 1; i < nodelist.size(); i++) {
                        auto const & cur_node = nodelist[i];
                        auto const & pre_node = nodelist[i-1];
                        totalLength += length(cur_node->pos - pre_node->pos);
                    }
                    float h = totalLength>0? 0.001*totalLength:1.0;

                    std::vector<float> vxs(nodelist.size());
                    std::vector<float> vys(nodelist.size());
                    std::vector<float> vzs(nodelist.size());
                    for(int i = 0; i < nodelist.size(); i++) {
                        auto const & cur_node = nodelist[i];
                        auto f = cur_node->userData().get2<float>("frame");
                        auto view = cur_node->view;
                        auto [x, y, z] = cur_node->pos;
                        auto [ux, uy, uz] = normalize(cur_node->up);
                        ts[i] = f;
                        xs[i] = x;
                        ys[i] = y;
                        zs[i] = z;
                        vxs[i] = view[0];
                        vys[i] = view[1];
                        vzs[i] = view[2];
                        xus[i] = x + h * ux;
                        yus[i] = y + h * uy;
                        zus[i] = z + h * uz;
                        fovs[i] = cur_node->fov;
                        apertures[i] = cur_node->aperture;
                        focalPlaneDistances[i] = cur_node->focalPlaneDistance;
                    }
                    curve_x = createCurvePoint(ts, xs);
                    curve_y = createCurvePoint(ts, ys);
                    curve_z = createCurvePoint(ts, zs);
                    curve_vx = createCurvePoint(ts, vxs);
                    curve_vy = createCurvePoint(ts, vys);
                    curve_vz = createCurvePoint(ts, vzs);
                    curve_ux = createCurvePoint(ts, xus);
                    curve_uy = createCurvePoint(ts, yus);
                    curve_uz = createCurvePoint(ts, zus);
                    curve_fov = createCurvePoint(ts, fovs);
                    curve_apertures = createCurvePoint(ts, apertures);
                    curve_fPD = createCurvePoint(ts, focalPlaneDistances);
                }
            }
            auto camera = std::make_unique<zeno::CameraObject>();
            camera->pos[0] = curve_x->eval(frameid);
            camera->pos[1] = curve_y->eval(frameid);
            camera->pos[2] = curve_z->eval(frameid);
            camera->fov = curve_fov->eval(frameid);
            camera->aperture = curve_apertures->eval(frameid);
            if (nodelist.size() == target_camera_count) {
                auto refUp = zeno::vec3f(curve_ux->eval(frameid),curve_uy->eval(frameid),curve_uz->eval(frameid));
                refUp = normalize(refUp);
                auto tarPos = zeno::vec3f(curve_tx->eval(frameid),curve_ty->eval(frameid),curve_tz->eval(frameid));
                camera->view = zeno::normalize(tarPos - camera->pos);
                auto cur_right = zeno::normalize(zeno::cross(camera->view, refUp));
                camera->up = zeno::normalize(zeno::cross(cur_right, camera->view));
                auto af = nodelist[0]->userData().get2<int>("AutoFocus", 1);
                if (af) {
                    camera->focalPlaneDistance = zeno::distance(camera->pos, tarPos);
                }
            }
            else {
                auto refUp = zeno::vec3f(curve_ux->eval(frameid),curve_uy->eval(frameid),curve_uz->eval(frameid)) - camera->pos;
                refUp = normalize(refUp);
                camera->view = zeno::vec3f(curve_vx->eval(frameid), curve_vy->eval(frameid), curve_vz->eval(frameid));
                camera->view = normalize(camera->view);
                auto cur_right = zeno::normalize(zeno::cross(camera->view, refUp));
                camera->up = zeno::normalize(zeno::cross(cur_right, camera->view));
                camera->focalPlaneDistance = curve_fPD->eval(frameid);
            }
            set_output("camera", std::move(camera));
        }

    }
};

ZENO_DEFNODE(CameraEval)({
    {
        {"frameid"},
        {"list", "nodelist"}
    },
    {
        {"CameraObject", "camera"},
    },
    {
    },
    {"FBX"},
});

struct ExtractCamera: zeno::INode {

    virtual void apply() override {
        auto cam = get_input2<zeno::CameraObject>("camobject");

        auto pos = std::make_shared<zeno::NumericObject>();
        auto up = std::make_shared<zeno::NumericObject>();
        auto view = std::make_shared<zeno::NumericObject>();
        auto fov = std::make_shared<zeno::NumericObject>();
        auto aperture = std::make_shared<zeno::NumericObject>();
        auto focalPlaneDistance = std::make_shared<zeno::NumericObject>();

        pos->set<zeno::vec3f>(cam->pos);
        up->set<zeno::vec3f>(cam->up);
        view->set<zeno::vec3f>(cam->view);
        fov->set<float>(cam->fov);
        aperture->set<float>(cam->aperture);
        focalPlaneDistance->set<float>(cam->focalPlaneDistance);


        set_output("pos", std::move(pos));
        set_output("up", std::move(up));
        set_output("view", std::move(view));
        set_output("fov", std::move(fov));
        set_output("aperture", std::move(aperture));
        set_output("focalPlaneDistance", std::move(focalPlaneDistance));
    }
};
ZENDEFNODE(ExtractCamera,
           {       /* inputs: */
               {
                    "camobject"
               },  /* outputs: */
               {
                   "pos", "up", "view", "fov", "aperture", "focalPlaneDistance"
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct DirtyTBN : INode {
    virtual void apply() override {

        auto AxisT = get_input2<zeno::vec3f>("T");
        auto AxisB = get_input2<zeno::vec3f>("B");
        //auto AxisN = get_input2<zeno::vec3f>("N");

        if (lengthSquared(AxisT) == 0 ) {
            AxisT = {1,0,0};
        }
        AxisT = zeno::normalize(AxisT);
        
        if (lengthSquared(AxisB) == 0 ) {
            AxisB = {0,0,1};
        }
        AxisB = zeno::normalize(AxisB);

        auto tmp = zeno::dot(AxisT, AxisB);
        if (abs(tmp) > 0.0) { // not vertical
            AxisB -= AxisT * tmp;
            AxisB = zeno::normalize(AxisB);
        }
        
        if (has_input("prim")) {
            auto prim = get_input<PrimitiveObject>("prim");

            auto pos = prim->userData().get2<zeno::vec3f>("pos", {0,0,0});
            auto scale = prim->userData().get2<zeno::vec3f>("scale", {1,1,1});

            auto v0 = pos - AxisT * scale[0] * 0.5f - AxisB * scale[2] * 0.5f;
            auto e1 = AxisT * scale[0];
            auto e2 = AxisB * scale[2];

            prim->verts[0] = v0 + e1 + e2;
            prim->verts[1] = v0 + e1;
            prim->verts[2] = v0 + e2;
            prim->verts[3] = v0;

            set_output("prim", std::move(prim));
        }
    }
};


ZENO_DEFNODE(DirtyTBN)({
    {
        {"PrimitiveObject", "prim"},
        {"vec3f", "T", "1, 0, 0"},
        {"vec3f", "B", "0, 0, 1"},
    },
    {
        "prim"
    },
    {},
    {"shader"},
});

struct LiveMeshNode : INode {
    typedef std::vector<std::vector<float>> UVS;
    typedef std::vector<std::vector<float>> VERTICES;
    typedef std::vector<int> VERTEX_COUNT;
    typedef std::vector<int> VERTEX_LIST;

    struct PrimIngredient{
        UVS uvs;
        VERTICES vertices;
        VERTEX_COUNT vertexCount;
        VERTEX_LIST vertexList;
    };

    void GeneratePrimitiveObject(PrimIngredient& ingredient, std::shared_ptr<zeno::PrimitiveObject> primObject){
        auto& vert = primObject->verts;
        auto& loops = primObject->loops;
        auto& polys = primObject->polys;

        for(int i=0; i<ingredient.vertices.size(); i++){
            auto& v = ingredient.vertices[i];
            vert.emplace_back(v[0], v[1], v[2]);
        }

        int start = 0;
        for(int i=0; i<ingredient.vertexCount.size(); i++){
            auto count = ingredient.vertexCount[i];
            for(int j=start; j<start+count; j++){
                loops.emplace_back(ingredient.vertexList[j]);
            }
            polys.emplace_back(start, count);

            start += count;
        }

        primObject->uvs.resize(loops.size());
        for (auto i = 0; i < loops.size(); i++) {
            primObject->uvs[i] = vec2f(ingredient.uvs[i][0], ingredient.uvs[i][1]);
        }
        auto& loopuvs = primObject->loops.add_attr<int>("uvs");
        for (auto i = 0; i < loops.size(); i++) {
            loopuvs[i] = i;
        }
    }

    virtual void apply() override {
        auto outDict = get_input2<bool>("outDict");
        auto prims_list = std::make_shared<zeno::ListObject>();
        auto prims_dict = std::make_shared<zeno::DictObject>();
        auto vertSrc = get_input2<std::string>("vertSrc");

        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<zeno::NumericObject>("frameid")->get<int>();
        } else {
            frameid = getGlobalState()->frameid;
        }

        if(! vertSrc.empty()){
            using json = nlohmann::json;

            json parseData = json::parse(vertSrc);

            /*
            auto& frameData = parseData[std::to_string(frameid)];
            auto frameDataSize = frameData["DATA"].size();

            std::cout << "src size " << vertSrc.size()
                      << " data size " << frameDataSize
                      << " frame " << frameid
                      << "\n";

            auto& AllMeshData = frameData["DATA"];
            for(auto& mapItem: AllMeshData.items()){
                auto prim = std::make_shared<zeno::PrimitiveObject>();
                std::cout << "iter map key " << mapItem.key() << "\n";
                auto& mapData = mapItem.value();
                int vertices_size = mapData["MESH_POINTS"].size();
                int vertexCount_size = mapData["MESH_VERTEX_COUNTS"].size();
                int vertexList_size = mapData["MESH_VERTEX_LIST"].size();
                PrimIngredient ingredient;
                ingredient.vertices = mapData["MESH_POINTS"].get<VERTICES>();
                ingredient.vertexCount = mapData["MESH_VERTEX_COUNTS"].get<VERTEX_COUNT>();
                ingredient.vertexList = mapData["MESH_VERTEX_LIST"].get<VERTEX_LIST>();
                std::cout << "Vertices Size " << vertices_size << " " << vertexCount_size << " " << vertexList_size << "\n";
                GeneratePrimitiveObject(ingredient, prim);

                prims->arr.emplace_back(prim);
            }
             */

            auto& parsedFrameData = parseData[std::to_string(frameid)];
            if(! parsedFrameData.empty()){
                auto bPathI = parsedFrameData["BPATHI"].get<std::string>();
                auto bPath = parsedFrameData["BPATH"].get<std::string>();
                std::cout<< "bPath info " << bPathI << "\n";
                std::ifstream t(bPathI);
                std::stringstream buffer;
                buffer << t.rdbuf();

                auto& sizesData = parsedFrameData["SIZES"];

                json infoData = json::parse(buffer.str());
                for(auto& mapItem: infoData.items()){
                    auto& key = mapItem.key();
                    auto& value = mapItem.value();

                    auto sizes = sizesData[key].get<std::vector<int>>();

                    auto u = value["UV"].get<std::string>();
                    auto v = value["VERTEX"].get<std::string>();
                    auto i = value["INDICES"].get<std::string>();
                    auto c = value["COUNTS"].get<std::string>();

                    std::cout << "sync info " << key << " sizes " << sizes.size() << "\n";
                    std::cout << " u " << u << "\n";
                    std::cout << " v " << v << "\n";
                    std::cout << " i " << i << "\n";
                    std::cout << " c " << c << "\n";

                    auto pu = bPath+"/"+u;
                    auto pv = bPath+"/"+v;
                    auto pi = bPath+"/"+i;
                    auto pc = bPath+"/"+c;

                    std::cout << " u.p " << pu << " u.s " << sizes[0] << "\n";
                    std::cout << " v.p " << pv << " v.s " << sizes[1] << "\n";
                    std::cout << " i.p " << pi << " i.s " << sizes[2] << "\n";
                    std::cout << " c.p " << pc << " c.s " << sizes[3] << "\n";

                    FILE *fp_u = fopen(pu.c_str(), "rb");
                    FILE *fp_v = fopen(pv.c_str(), "rb");
                    FILE *fp_i = fopen(pi.c_str(), "rb");
                    FILE *fp_c = fopen(pc.c_str(), "rb");

                    float *_u = new float[sizes[0]];
                    float *_v = new float[sizes[1]];
                    int   *_i = new int[sizes[2]];
                    int   *_c = new int[sizes[3]];

                    fread((void*)(_u), sizeof(float), sizes[0], fp_u);
                    fread((void*)(_v), sizeof(float), sizes[1], fp_v);
                    fread((void*)(_i), sizeof(int), sizes[2], fp_i);
                    fread((void*)(_c), sizeof(int), sizes[3], fp_c);

                    UVS _vu{};
                    VERTICES _vv{};
                    VERTEX_LIST _vi{};
                    VERTEX_COUNT _vc{};

                    for(int s = 0; s < sizes[0]; s+=2){
                        _vu.push_back({_u[s], _u[s+1]});
                    }
                    for(int s = 0; s < sizes[1]; s+=3){
                        _vv.push_back({_v[s], _v[s+1], _v[s+2]});
                    }
                    for(int s = 0; s < sizes[2]; ++s){
                        _vi.push_back(_i[s]);
                    }
                    for(int s = 0; s < sizes[3]; ++s){
                        _vc.push_back(_c[s]);
                    }

                    delete [] _u;
                    delete [] _v;
                    delete [] _c;
                    delete [] _i;

                    fclose(fp_u);
                    fclose(fp_v);
                    fclose(fp_i);
                    fclose(fp_c);

                    PrimIngredient ingredient;
                    ingredient.uvs = _vu;
                    ingredient.vertices = _vv;
                    ingredient.vertexCount = _vc;
                    ingredient.vertexList = _vi;
                    auto prim = std::make_shared<zeno::PrimitiveObject>();
                    GeneratePrimitiveObject(ingredient, prim);
                    if(outDict) {
                        prims_dict->lut[key] = prim;
                    }else{
                        prims_list->arr.emplace_back(prim);
                    }
                }
            }else{
                std::cout << "not parsed frame " << frameid << "\n";
            }
        }
        if(outDict) {
            set_output("prims", std::move(prims_dict));
        }else{
            set_output("prims", std::move(prims_list));
        }
    }
};

ZENO_DEFNODE(LiveMeshNode)({
    {
        {"frameid"},
        {"string", "vertSrc", ""},
        {"bool", "outDict", "false"}
    },
    {
        "prims"
    },
    {
    },
    {"FBX"},
});


struct LiveCameraNode : INode{
    typedef std::vector<float> CAMERA_TRANS;
    struct CameraIngredient{
        CAMERA_TRANS translation;
    };

    virtual void apply() override {
        auto camera = std::make_shared<zeno::CameraObject>();
        auto camSrc = get_input2<std::string>("camSrc");

        if(! camSrc.empty()){
            std::cout << "src came " << camSrc.size() << "\n";
            using json = nlohmann::json;
            json parseData = json::parse(camSrc);
            int translation_size = parseData["translation"].size();
            CameraIngredient ingredient;
            ingredient.translation = parseData["translation"].get<CAMERA_TRANS>();
            std::cout << " translation_size " << translation_size << "\n";

            float transX = ingredient.translation[0];
            float transY = ingredient.translation[1];
            float transZ = ingredient.translation[2];
            float rotateX = ingredient.translation[3];
            float rotateY = ingredient.translation[4];
            float rotateZ = ingredient.translation[5];
            //float scaleX = ingredient.translation[6];
            //float scaleY = ingredient.translation[7];
            //float scaleZ = ingredient.translation[8];

            glm::mat4 transMatrixR = glm::translate(glm::vec3(transX, transY, -transZ));
            glm::mat4 transMatrixL = glm::translate(glm::vec3(transX, transY, transZ));
            float ax = rotateX * (M_PI / 180.0);
            float ay = rotateY * (M_PI / 180.0);
            float az = rotateZ * (M_PI / 180.0);
            glm::mat3 mx = glm::mat3(1,0,0,  0,cos(ax),-sin(ax),  0,sin(ax),cos(ax));
            glm::mat3 my = glm::mat3(cos(ay),0,sin(ay),  0,1,0,  -sin(ay),0,cos(ay));
            glm::mat3 mz = glm::mat3(cos(az),-sin(az),0,  sin(az),cos(az),0,  0,0,1);
            auto rotateMatrix3 = mx*my*mz;
            auto rotateMatrix4 = glm::mat4((rotateMatrix3));

            //auto matrix = transMatrixL * rotateMatrix4 * transMatrixR;
            auto matrix = rotateMatrix4;
            glm::vec3 trans, scale, skew; glm::quat rot; glm::vec4 perp;
            glm::decompose(matrix, trans, rot, scale, skew, perp);
            glm::mat3 rotMatrix = glm::mat3_cast(rot);

            camera->pos = zeno::vec3f(transX, transY, transZ);
            camera->view = zeno::vec3f(rotMatrix[2][0], rotMatrix[2][1], rotMatrix[2][2]);
            camera->up = zeno::vec3f(rotMatrix[1][0], rotMatrix[1][1], rotMatrix[1][2]);
            std::cout << "RotateMatrix\n\t" << rotMatrix[0][0] << " " << rotMatrix[0][1] << " " << rotMatrix[0][2]
                      << "\n\t" << rotMatrix[1][0] << " " << rotMatrix[1][1] << " " << rotMatrix[1][2]
                      << "\n\t" << rotMatrix[2][0] << " " << rotMatrix[2][1] << " " << rotMatrix[2][2] << "\n";
            std::cout << "pos " <<  trans[0] << " " << trans[1] << " " << trans[2] << "\n";
            std::cout << "view " <<  camera->view[0] << " " << camera->view[1] << " " << camera->view[2] << "\n";
            std::cout << "up " <<  camera->up[0] << " " << camera->up[1] << " " << camera->up[2] << "\n";
        }
        set_output("camera", std::move(camera));
    }
};

//ZENO_DEFNODE(LiveCameraNode)({
//    {
//        {"string", "camSrc", ""},
//    },
//    {
//        "camera"
//    },
//    {
//    },
//    {"FBX"},
//});

}
}
