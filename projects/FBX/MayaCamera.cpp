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

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

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
        camera->fnear = get_input2<float>("frame");

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(CameraNode)({
    {
        {"vec3f", "pos", "0,0,5"},
        {"vec3f", "up", "0,1,0"},
        {"vec3f", "view", "0,0,-1"},
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
        std::string inter_mode;
        auto inter = get_param<std::string>("inter");
        if (inter == "Bezier"){
            inter_mode = "Bezier";
        }else if(inter == "Linear"){
            inter_mode = "Linear";
        }

        //zeno::log_info("CameraEval frame {}", frameid);

        // TODO sort CameraObject by fnear(frameId)

        if(nodelist.size() == 1){
            auto n = nodelist[0];
            out_pos->set(n->pos);
            out_up->set(n->up);
            out_view->set(n->view);
            //zeno::log_info("CameraEval size 1");
        }else{
            auto fn = nodelist[0];
            auto en = nodelist[nodelist.size()-1];
            if(frameid <= (int)fn->fnear){
                out_pos->set(fn->pos);
                out_up->set(fn->up);
                out_view->set(fn->view);
                //zeno::log_info("CameraEval first frame");
            }else if(frameid >= (int)en->fnear) {
                out_pos->set(en->pos);
                out_up->set(en->up);
                out_view->set(en->view);
                //zeno::log_info("CameraEval last frame");
            }else{
                for(int i=1;i<nodelist.size();i++){
                    auto const & n = nodelist[i];
                    auto const & nm = nodelist[i-1];
                    if(frameid < (int)n->fnear){
                        zeno::vec3f pos;
                        zeno::vec3f up;
                        zeno::vec3f view;
                        float factor = (float)(frameid - (int)nm->fnear) / ((int)n->fnear - (int)nm->fnear);

                        //zeno::log_info("CameraEval Interval {} {} factor {}", (int)nm->fnear, (int)n->fnear, factor);

                        // linear interpolation
                        if(inter_mode == "Linear"){
                            pos = n->pos * factor + nm->pos*(1.0f-factor);
                            up = n->up * factor + nm->up*(1.0f-factor);
                            view = n->view * factor + nm->view*(1.0f-factor);

                        }
                        // Bezier interpolation
                        else if(inter_mode == "Bezier"){
                            struct BezierCompute{
                                static zeno::vec3f compute(zeno::vec3f p1, zeno::vec3f p2, float t){
                                    return (1-t)*p1+t*p2;
                                }

                                static zeno::vec3f bezier( std::vector<zeno::vec3f> const&p, float t ){
                                    std::vector<zeno::vec3f> ps = p;
                                    auto iter = ps.size();
                                    for(int z=0; z<iter; z++){
                                        auto n=ps.size();
                                        std::vector<zeno::vec3f> tmp;
                                        for(int i=0;i<n-1;i++){
                                            auto cr = zeno::vec3f(compute(ps[i], ps[i+1], t));
                                            tmp.push_back(cr);
                                        }
                                        ps=tmp;
                                        iter--;
                                    }
                                    return compute(ps[0], ps[1], t);
                                }
                            };

                            int frame_diff = std::abs((int)n->fnear - (int)nm->fnear);
                            float percent = frame_diff <= 2 ? 0.5f : 2.0f/frame_diff;

                            float pos_x_diff = std::abs(n->pos[0] - nm->pos[0]);
                            float pos_y_diff = std::abs(n->pos[1] - nm->pos[1]);
                            float pos_z_diff = std::abs(n->pos[2] - nm->pos[2]);
                            float pxs = (n->pos[0] - nm->pos[0]) > 0?1.0f:-1.0f;
                            float pys = (n->pos[1] - nm->pos[1]) > 0?1.0f:-1.0f;
                            float pzs = (n->pos[2] - nm->pos[2]) > 0?1.0f:-1.0f;
                            float up_x_diff = std::abs(n->up[0] - nm->up[0]);
                            float up_y_diff = std::abs(n->up[1] - nm->up[1]);
                            float up_z_diff = std::abs(n->up[2] - nm->up[2]);
                            float uxs = (n->up[0] - nm->up[0]) > 0?1.0f:-1.0f;
                            float uys = (n->up[1] - nm->up[1]) > 0?1.0f:-1.0f;
                            float uzs = (n->up[2] - nm->up[2]) > 0?1.0f:-1.0f;
                            float view_x_diff = std::abs(n->view[0] - nm->view[0]);
                            float view_y_diff = std::abs(n->view[1] - nm->view[1]);
                            float view_z_diff = std::abs(n->view[2] - nm->view[2]);
                            float vxs = (n->view[0] - nm->view[0]) > 0?1.0f:-1.0f;
                            float vys = (n->view[1] - nm->view[1]) > 0?1.0f:-1.0f;
                            float vzs = (n->view[2] - nm->view[2]) > 0?1.0f:-1.0f;

                            // TODO The control points consider the front and back frame trends
                            std::vector<zeno::vec3f> tp_pos;
                            std::vector<zeno::vec3f> tp_up;
                            std::vector<zeno::vec3f> tp_view;
                            zeno::vec3f diff_pos = {percent*pos_x_diff*pxs, percent*pos_y_diff*pys, percent*pos_z_diff*pzs};
                            zeno::vec3f diff_up = {percent*up_x_diff*uxs, percent*up_y_diff*uys, percent*up_z_diff*uzs};
                            zeno::vec3f diff_view = {percent*view_x_diff*vxs, percent*view_y_diff*vys, percent*view_z_diff*vzs};

                            tp_pos.push_back(nm->pos);
                            tp_pos.push_back(nm->pos + diff_pos);
                            tp_pos.push_back(n->pos);
                            tp_pos.push_back(n->pos - diff_pos);
                            auto p = BezierCompute::bezier(tp_pos, factor);
                            tp_up.push_back(nm->up);
                            tp_up.push_back(nm->up + diff_up);
                            tp_up.push_back(n->up);
                            tp_up.push_back(n->up - diff_up);
                            auto u = BezierCompute::bezier(tp_up, factor);
                            tp_view.push_back(nm->view);
                            tp_view.push_back(nm->view + diff_view);
                            tp_view.push_back(n->view);
                            tp_view.push_back(n->view - diff_view);
                            auto v = BezierCompute::bezier(tp_view, factor);
                            //zeno::log_info("Inter FrameDiff {} Percent {}", frame_diff, percent);
                            //zeno::log_info("DiffPos {} {} {}", diff_pos[0], diff_pos[1], diff_pos[2]);
                            //zeno::log_info("DiffUp {} {} {}", diff_up[0], diff_up[1], diff_up[2]);
                            //zeno::log_info("DiffView {} {} {}", diff_view[0], diff_view[1], diff_view[2]);
                            pos = p;
                            up = u;
                            view = v;
                        }

                        //zeno::log_info("Inter Pos {} {} {}", pos[0], pos[1], pos[2]);
                        //zeno::log_info("Inter Up {} {} {}", up[0], up[1], up[2]);
                        //zeno::log_info("Inter View {} {} {}", view[0], view[1], view[2]);

                        out_pos->set(pos);
                        out_up->set(up);
                        out_view->set(view);

                        break;
                    }
                }
            }
        }

        set_output("pos", std::move(out_pos));
        set_output("up", std::move(out_up));
        set_output("view", std::move(out_view));
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
            tris.emplace_back(zeno::vec3i(2, 3, 0));

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
                int tmp = prim->tris[i][2];
                prim->tris[i][2] = prim->tris[i][0];
                prim->tris[i][0] = tmp;
            }
        }

        prim->userData().setLiterial("isL", std::move(isL));
        prim->userData().setLiterial("ivD", std::move(inverdir));
        prim->userData().setLiterial("pos", std::move(position));
        prim->userData().setLiterial("scale", std::move(scale));
        prim->userData().setLiterial("rotate", std::move(rotate));
        prim->userData().setLiterial("shape", std::move(shape));

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
        {"int", "islight", "1"},
        {"int", "invertdir", "1"}
    },
    {
        "prim"
    },
    {
        {"enum Disk Plane", "Shape", "Plane"},
    },
    {"FBX"},
});

}
}
