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

#include "assimp/scene.h"

#include "Definition.h"
#include "json.hpp"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <fstream>
#include <regex>

#define SET_CAMERA_DATA                         \
    out_pos->set(n->pos);                       \
    out_up->set(n->up);                         \
    out_view->set(n->view);                     \
    out_fov->set(n->fov);                       \
    out_aperture->set(n->aperture);             \
    out_focalPlaneDistance->set(n->focalPlaneDistance); \

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

struct CameraNode: zeno::INode{
    virtual void apply() override {
        auto camera = std::make_unique<zeno::CameraObject>();

        camera->pos = get_input2<zeno::vec3f>("pos");
        camera->up = get_input2<zeno::vec3f>("up");
        camera->view = get_input2<zeno::vec3f>("view");
        camera->fov = get_input2<float>("fov");
        camera->aperture = get_input2<float>("aperture");
        camera->focalPlaneDistance = get_input2<float>("focalPlaneDistance");
        camera->userData().set2("frame", get_input2<float>("frame"));

        auto other_props = get_input2<std::string>("other");
        std::regex reg(",");
        std::sregex_token_iterator p(other_props.begin(), other_props.end(), reg, -1);
        std::sregex_token_iterator end;
        std::vector<float> prop_vals;
        while (p != end) {
            prop_vals.push_back(std::stof(*p));
            p++;
        }
        if (prop_vals.size() == 6) {
            camera->center = {prop_vals[0], prop_vals[1], prop_vals[2]};
            camera->theta = prop_vals[3];
            camera->phi = prop_vals[4];
            camera->radius = prop_vals[5];
        }

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(CameraNode)({
    {
        {"vec3f", "pos", "0,0,5"},
        {"vec3f", "up", "0,1,0"},
        {"vec3f", "view", "0,0,-1"},
        {"float", "fov", "45"},
        {"float", "aperture", "0.1"},
        {"float", "focalPlaneDistance", "2.0"},
        {"string", "other", ""},
        {"int", "frame", "0"},
    },
    {
        {"CameraObject", "camera"},
    },
    {
    },
    {"FBX"},
});

struct CameraEval: zeno::INode {
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<zeno::NumericObject>("frameid")->get<int>();
        } else {
            frameid = getGlobalState()->frameid;
        }

        auto nodelist = get_input<zeno::ListObject>("nodelist")->get<zeno::CameraObject>();

        auto out_pos = std::make_unique<zeno::NumericObject>();
        auto out_up = std::make_unique<zeno::NumericObject>();
        auto out_view = std::make_unique<zeno::NumericObject>();
        auto out_fov = std::make_unique<zeno::NumericObject>();
        auto out_aperture = std::make_unique<zeno::NumericObject>();
        auto out_focalPlaneDistance = std::make_unique<zeno::NumericObject>();
        std::string inter_mode;
        auto inter = get_param<std::string>("inter");
        if (inter == "Bezier"){
            inter_mode = "Bezier";
        }else if(inter == "Linear"){
            inter_mode = "Linear";
        }

        //zeno::log_info("CameraEval frame {}", frameid);

        // TODO sort CameraObject by (frameId)

        if(nodelist.size() == 1){
            auto n = nodelist[0];
            SET_CAMERA_DATA
            //zeno::log_info("CameraEval size 1");
        }else{
            int ff = (int)nodelist[0]->userData().get2<float>("frame");
            int lf = (int)nodelist[nodelist.size()-1]->userData().get2<float>("frame");
            if(frameid <= ff){
                auto n = nodelist[0];
                SET_CAMERA_DATA
                //zeno::log_info("CameraEval first frame");
            }else if(frameid >= lf) {
                auto n = nodelist[nodelist.size()-1];
                SET_CAMERA_DATA
                //zeno::log_info("CameraEval last frame");
            }else{
                for(int i=1;i<nodelist.size();i++){
                    auto const & n = nodelist[i];
                    auto const & nm = nodelist[i-1];
                    int cf = (int)n->userData().get2<float>("frame");
                    if(frameid < cf){
                        zeno::vec3f pos;
                        zeno::vec3f up;
                        zeno::vec3f view;
                        float fov;
                        float aperture;
                        float focalPlaneDistance;

                        float factor = (float)(frameid - (int)nm->userData().get2<float>("frame"))
                                       / (float)((int)n->userData().get2<float>("frame")
                                           - (int)nm->userData().get2<float>("frame"));

                        // linear interpolation
                        if(inter_mode == "Linear"){
                            pos = n->pos * factor + nm->pos*(1.0f-factor);
                            up = n->up * factor + nm->up*(1.0f-factor);
                            view = n->view * factor + nm->view*(1.0f-factor);
                            fov = n->fov * factor + nm->fov*(1.0f-factor);
                            aperture = n->aperture * factor + nm->aperture*(1.0f-factor);
                            focalPlaneDistance = n->focalPlaneDistance * factor + nm->focalPlaneDistance*(1.0f-factor);

                        }
                        // Bezier interpolation
                        else if(inter_mode == "Bezier"){
                            // TODO The control points consider the front and back frame trends

                            float c1of = 0.4f;
                            float c2of = 0.6f;

                            pos = BezierCompute::compute(c1of, c2of, factor, n->pos, nm->pos);
                            up = BezierCompute::compute(c1of, c2of, factor, n->up, nm->up);
                            view = BezierCompute::compute(c1of, c2of, factor, n->view, nm->view);
                            fov = BezierCompute::compute(c1of, c2of, factor, n->fov, nm->fov);
                            aperture = BezierCompute::compute(c1of, c2of, factor, n->aperture, nm->aperture);
                            focalPlaneDistance = BezierCompute::compute(c1of, c2of, factor, n->focalPlaneDistance, nm->focalPlaneDistance);
                        }

                        //zeno::log_info("Inter Pos {} {} {}", pos[0], pos[1], pos[2]);
                        //zeno::log_info("Inter Up {} {} {}", up[0], up[1], up[2]);
                        //zeno::log_info("Inter View {} {} {}", view[0], view[1], view[2]);

                        out_pos->set(pos);
                        out_up->set(up);
                        out_view->set(view);
                        out_fov->set(fov);
                        out_aperture->set(aperture);
                        out_focalPlaneDistance->set(focalPlaneDistance);
                        break;
                    }
                }
            }
        }

        set_output("pos", std::move(out_pos));
        set_output("up", std::move(out_up));
        set_output("view", std::move(out_view));
        set_output("fov", std::move(out_fov));
        set_output("aperture", std::move(out_aperture));
        set_output("focalPlaneDistance", std::move(out_focalPlaneDistance));
    }
};

ZENO_DEFNODE(CameraEval)({
    {
        {"frameid"},
        {"nodelist"}
    },
    {
        {"vec3f", "pos"},
        {"vec3f", "up"},
        {"vec3f", "view"},
        {"float", "fov"},
        {"float", "aperture"},
        {"float", "focalPlaneDistance"},
    },
    {
        {"enum Bezier Linear ", "inter", "Bezier"},
    },
    {"FBX"},
});

struct LightNode : INode {
    virtual void apply() override {
        auto isL = get_input2<int>("islight");
        auto inverdir = get_input2<int>("invertdir");
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto intensity = get_input2<float>("intensity");
        auto color = get_input2<zeno::vec3f>("color");
        auto shapeParam = get_param<std::string>("Shape");
        std::string shape;
        if (shapeParam == "Disk"){
            shape = "Disk";
        }else if(shapeParam == "Plane"){
            shape = "Plane";
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &verts = prim->verts;
        auto &tris = prim->tris;

        // Rotate
        float ax = rotate[0] * (3.14159265358979323846 / 180.0);
        float ay = rotate[1] * (3.14159265358979323846 / 180.0);
        float az = rotate[2] * (3.14159265358979323846 / 180.0);
        glm::mat3 mx = glm::mat3(1, 0, 0, 0, cos(ax), -sin(ax), 0, sin(ax), cos(ax));
        glm::mat3 my = glm::mat3(cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay));
        glm::mat3 mz = glm::mat3(cos(az), -sin(az), 0, sin(az), cos(az), 0, 0, 0, 1);

        if(shape == "Plane"){
            auto start_point = zeno::vec3f(0.5, 0, 0.5);
            float rm = 1.0f;
            float cm = 1.0f;

            // Plane Verts
            for(int i=0; i<=1; i++){
                auto rp = start_point - zeno::vec3f(i*rm, 0, 0);
                for(int j=0; j<=1; j++){
                    auto p = rp - zeno::vec3f(0, 0, j*cm);
                    // S R T
                    p = p * scale;
                    auto gp = glm::vec3(p[0], p[1], p[2]);
                    gp = mz * my * mx * gp;
                    p = zeno::vec3f(gp.x, gp.y, gp.z);
                    auto zcp = zeno::vec3f(p[0], p[1], p[2]);
                    zcp = zcp + position;

                    verts.push_back(zcp);
                }
            }

            // Plane Indices
            tris.emplace_back(zeno::vec3i(0, 3, 1));
            tris.emplace_back(zeno::vec3i(3, 0, 2));

        }else if(shape == "Disk"){
            int divisions = 13;
            verts.emplace_back(zeno::vec3f(0, 0, 0)+position);

            for (int i = 0; i < divisions; i++) {
                float rad = 2 * 3.14159265358979323846 * i / divisions;
                auto p = zeno::vec3f(cos(rad), 0, -sin(rad));
                // S R T
                p = p * scale;
                auto gp = glm::vec3(p[0], p[1], p[2]);
                gp = mz * my * mx * gp;
                p = zeno::vec3f(gp.x, gp.y, gp.z);
                p+= position;

                verts.emplace_back(p);
                tris.emplace_back(i+1, 0, i+2);
            }
            tris[tris.size()-1] = zeno::vec3i(divisions, 0, 1);
        }

        auto &clr = prim->verts.add_attr<zeno::vec3f>("clr");
        auto c = color * intensity;
        for(int i=0; i<verts.size(); i++){
            clr[i] = c;
        }

        if(inverdir){
            for(int i=0;i<prim->tris.size(); i++){
                int tmp = prim->tris[i][1];
                prim->tris[i][1] = prim->tris[i][0];
                prim->tris[i][0] = tmp;
            }
        }

        prim->userData().set2("isRealTimeObject", std::move(isL));
        prim->userData().set2("isL", std::move(isL));
        prim->userData().set2("ivD", std::move(inverdir));
        prim->userData().set2("pos", std::move(position));
        prim->userData().set2("scale", std::move(scale));
        prim->userData().set2("rotate", std::move(rotate));
        prim->userData().set2("shape", std::move(shape));
        prim->userData().set2("color", std::move(color));
        prim->userData().set2("intensity", std::move(intensity));

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(LightNode)({
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vec3f", "color", "1, 1, 1"},
        {"float", "intensity", "1"},
        {"bool", "islight", "1"},
        {"bool", "invertdir", "1"}
    },
    {
        "prim"
    },
    {
        {"enum Disk Plane", "Shape", "Plane"},
    },
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
