#include <zeno/zeno.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/CurveObject.h>

namespace zeno {

struct MakeCamera : INode {
    virtual void apply() override {
        auto camera = std::make_unique<CameraObject>();

        camera->pos = get_input2<vec3f>("pos");
        camera->up = get_input2<vec3f>("up");
        camera->view = get_input2<vec3f>("view");
        camera->ffar = get_input2<float>("far");
        camera->fnear = get_input2<float>("near");
        camera->fov = get_input2<float>("fov");

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(MakeCamera)({
    {
        {"vec3f", "pos", "0,0,5"},
        {"vec3f", "up", "0,1,0"},
        {"vec3f", "view", "0,0,-1"},
        {"float", "near", "0.01"},
        {"float", "far", "20000"},
        {"float", "fov", "45"},
    },
    {
        {"CameraObject", "camera"},
    },
    {
    },
    {"shader"},
});

struct CameraNode: INode{
    virtual void apply() override {
        auto camera = std::make_unique<CameraObject>();

        camera->pos = get_input2<vec3f>("pos");
        camera->up = get_input2<vec3f>("up");
        camera->view = get_input2<vec3f>("view");
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
    {"shader"},
});

struct CameraEval: INode {
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
                                static vec3f compute(vec3f p1, vec3f p2, float t){
                                    return (1-t)*p1+t*p2;
                                }

                                static vec3f bezier( std::vector<vec3f> const&p, float t ){
                                    std::vector<vec3f> ps = p;
                                    auto iter = ps.size();
                                    for(int z=0; z<iter; z++){
                                        auto n=ps.size();
                                        std::vector<vec3f> tmp;
                                        for(int i=0;i<n-1;i++){
                                            auto cr = vec3f(compute(ps[i], ps[i+1], t));
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
                            std::vector<vec3f> tp_pos;
                            std::vector<vec3f> tp_up;
                            std::vector<vec3f> tp_view;
                            vec3f diff_pos = {percent*pos_x_diff*pxs, percent*pos_y_diff*pys, percent*pos_z_diff*pzs};
                            vec3f diff_up = {percent*up_x_diff*uxs, percent*up_y_diff*uys, percent*up_z_diff*uzs};
                            vec3f diff_view = {percent*view_x_diff*vxs, percent*view_y_diff*vys, percent*view_z_diff*vzs};

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
    {"shader"},
});

struct MakeLight : INode {
    virtual void apply() override {
        auto light = std::make_unique<LightObject>();
        light->lightDir = normalize(get_input2<vec3f>("lightDir"));
        light->intensity = get_input2<float>("intensity");
        light->shadowTint = get_input2<vec3f>("shadowTint");
        light->lightHight = get_input2<float>("lightHight");
        light->shadowSoftness = get_input2<float>("shadowSoftness");
        light->lightColor = get_input2<vec3f>("lightColor");
        light->lightScale = get_input2<float>("lightScale");
        light->isEnabled = get_input2<bool>("isEnabled");
        set_output("light", std::move(light));
    }
};

ZENO_DEFNODE(MakeLight)({
    {
        {"vec3f", "lightDir", "1,1,0"},
        {"float", "intensity", "10"},
        {"vec3f", "shadowTint", "0.2,0.2,0.2"},
        {"float", "lightHight", "1000.0"},
        {"float", "shadowSoftness", "1.0"},
        {"vec3f", "lightColor", "1,1,1"},
        {"float", "lightScale", "1"},
        {"bool", "isEnabled", "1"},
    },
    {
        {"LightObject", "light"},
    },
    {
    },
    {"shader"},
});

};
