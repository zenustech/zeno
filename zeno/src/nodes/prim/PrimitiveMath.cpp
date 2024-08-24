#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace {
using namespace zeno;

struct PlaneProjectPrimitive2DAABB : INode {
    virtual void apply() override {
        auto origin = get_input<NumericObject>("origin")->get<zeno::vec3f>();
        auto normal = get_input<NumericObject>("normal")->get<zeno::vec3f>();
        auto tangent = get_input<NumericObject>("tangent")->get<zeno::vec3f>();
        auto bitangent = get_input<NumericObject>("bitangent")->get<zeno::vec3f>();
        auto prim = get_input<PrimitiveObject>("prim");

        zeno::vec2f bmin(+1e6), bmax(-1e6);
        auto &pos = prim->attr<zeno::vec3f>("pos");
        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            auto p = pos[line[0]], q = pos[line[1]];
            auto u = dot(origin - p, normal);
            auto v = dot(q - p, normal);
            if (std::fabs(v) < 1e-6)
                continue;
            auto d = u / v;
            if (0 <= d && d <= 1) {
                auto dist = p + (q - p) * (u / v) - origin;
                zeno::vec2f coor{dot(dist, tangent), dot(dist, bitangent)};
                bmin = zeno::min(bmin, coor);
                bmax = zeno::max(bmax, coor);
            }
        }
        set_output("boundMin2D", std::make_shared<NumericObject>(bmin));
        set_output("boundMax2D", std::make_shared<NumericObject>(bmax));
    }
};

ZENDEFNODE(PlaneProjectPrimitive2DAABB, {
    {
        {gParamType_Vec3f, "normal", "", zeno::Socket_ReadOnly},
        {gParamType_Vec3f, "tangent", "", zeno::Socket_ReadOnly},
        {gParamType_Vec3f, "origin", "", zeno::Socket_ReadOnly},
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Vec3f, "bitangent", "", zeno::Socket_ReadOnly},
    },
    {{gParamType_Vec2f, "boundMin2D"}, {gParamType_Vec2f, "boundMax2D"}},
    {},
    {"math"},
});
}
