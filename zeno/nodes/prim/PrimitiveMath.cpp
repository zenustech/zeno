#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/ListObject.h>

using namespace zeno;

struct PlaneProjectPrimitive2DAABB : INode {
    virtual void apply() override {
        auto origin = get_input<NumericObject>("origin")->get<vec3f>();
        auto normal = get_input<NumericObject>("normal")->get<vec3f>();
        auto tangent = get_input<NumericObject>("tangent")->get<vec3f>();
        auto bitangent = get_input<NumericObject>("bitangent")->get<vec3f>();
        auto prim = get_input<PrimitiveObject>("prim");

        vec2f bmin(+1e6), bmax(-1e6);
        auto &pos = prim->attr<vec3f>("pos");
        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            auto p = pos[line[0]], q = pos[line[1]];
            auto u = dot(origin - p, normal);
            auto v = dot(q - p, normal);
            if (0 < u * v && u < v) {  // 0 < u / v && u / v < 1
                auto dist = p + (q - p) * (u / v) - origin;
                vec2f coor{dot(dist, tangent), dot(dist, bitangent)};
                bmin = zeno::min(bmin, coor);
                bmax = zeno::max(bmax, coor);
            }
        }
        set_output("boundMin2D", std::make_shared<NumericObject>(bmin));
        set_output("boundMax2D", std::make_shared<NumericObject>(bmax));
    }
};

ZENDEFNODE(PlaneProjectPrimitive2DAABB, {
    {"origin", "normal", "tangent", "bitangent", "prim"},
    {"boundMin2D", "boundMax2D"},
    {},
    {"mathematica"},
});


struct PlaneEvaluate2DCoorTo3D : INode {
    virtual void apply() override {
        auto origin = get_input<NumericObject>("origin")->get<vec3f>();
        auto tangent = get_input<NumericObject>("tangent")->get<vec3f>();
        auto bitangent = get_input<NumericObject>("bitangent")->get<vec3f>();
        auto coor = get_input<NumericObject>("coor2D")->get<vec2f>();

        auto pos = origin + coor[0] * tangent + coor[1] * bitangent;
        set_output("coor3D", std::make_shared<NumericObject>(pos));
    }
};

ZENDEFNODE(PlaneEvaluate2DCoorTo3D, {
    {"coor2D", "origin", "tangent", "bitangent"},
    {"coor3D"},
    {},
    {"mathematica"},
});
