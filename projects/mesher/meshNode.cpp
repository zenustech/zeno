#include <type_traits>
#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include "Render.h"
namespace zeno{
struct MPMMesher : IObject {
    Render* data;
};

struct MakeMPMMehser : INode {
    virtual void apply() override {
        auto mesher = IObject::make<MPMMesher>();
        auto smooth_iter = get_input("smooth_iter")->as<zeno::NumericObject>()->get<int>();
        auto start = get_input("start")->as<zeno::NumericObject>()->get<int>();
        auto end = get_input("end")->as<zeno::NumericObject>()->get<int>();
        auto edge_stretch = get_input("edge_stretch")->as<zeno::NumericObject>()->get<float>();
        auto path = get_input("pathPreFix")->as<zeno::StringObject>()->get();
        auto vtk_path = get_input("tet_path")->as<zeno::StringObject>()->get();
        mesher->data = new Render();
        mesher->data->max_smooth_iter_bound = std::min(smooth_iter, 20);
        mesher->data->edge_strech_threshold = edge_stretch;
        mesher->data->start_frame = start;
        mesher->data->end_frame = end;
        mesher->data->m_vtk_path = vtk_path;
        mesher->data->inputpath = path;
        mesher->data->preprocess();
        set_output("MPMMehser", mesher);
    }
};
ZENDEFNODE(MakeMPMMehser, {
    {"pathPreFix", "tet_path","smooth_iter", "start", "end",  "edge_stretch"},
    {"MPMMehser"},
    {},
    {"GPUMPM"},
});

struct MesherProcessFrame : INode {
    virtual void apply() override {
        
        auto mesher = get_input("Mesher")->as<MPMMesher>();
        auto outPrim = IObject::make<PrimitiveObject>();
        auto i = get_input("frameNumber")->as<NumericObject>()->get<int>();
        mesher->data->process(i);
        outPrim->resize(mesher->data->vertices.size());
        auto &pos = outPrim->add_attr<vec3f>("pos");
        #pragma omp parallel for
        for(int64_t i=0;i<mesher->data->vertices.size();i++)
        {
            pos[i] = zeno::vec3f(mesher->data->vertices[i][0], mesher->data->vertices[i][1], mesher->data->vertices[i][2]);
        }
        outPrim->tris.resize(mesher->data->tris.size());
        #pragma omp parallel for
        for(int64_t i=0;i<mesher->data->tris.size();i++)
        {
            outPrim->tris[i] = zeno::vec3i(mesher->data->tris[i][0], mesher->data->tris[i][1], mesher->data->tris[i][2]);
        }
        //mesher->data
        set_output("FramePrimitive", outPrim);
    }
};
ZENDEFNODE(MesherProcessFrame, {
    {"Mesher", "frameNumber"},
    {"FramePrimitive"},
    {},
    {"GPUMPM"},
});
}