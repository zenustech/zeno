#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct MakeVisualAABBPrimitive : INode {
    virtual void apply() override {
        auto a = has_input("boundMin")
            ? get_input<NumericObject>("boundMin")->get<vec3f>()
            : vec3f(-1, -1, -1);
        auto b = has_input("boundMax")
            ? get_input<NumericObject>("boundMax")->get<vec3f>()
            : vec3f(+1, +1, +1);
        auto connType = get_param<int>("connType");

        auto prim = std::make_shared<PrimitiveObject>();
        auto &pos = prim->add_attr<vec3f>("pos");

        prim->resize(8);
        pos[0] = vec3f(a[0], a[1], a[2]);
        pos[1] = vec3f(b[0], a[1], a[2]);
        pos[2] = vec3f(b[0], b[1], a[2]);
        pos[3] = vec3f(a[0], b[1], a[2]);
        pos[4] = vec3f(a[0], a[1], b[2]);
        pos[5] = vec3f(b[0], a[1], b[2]);
        pos[6] = vec3f(b[0], b[1], b[2]);
        pos[7] = vec3f(a[0], b[1], b[2]);

        if (connType == 2) {
            prim->lines.resize(12);
            prim->lines[0] = vec2i(0, 1);
            prim->lines[1] = vec2i(1, 2);
            prim->lines[2] = vec2i(2, 3);
            prim->lines[3] = vec2i(3, 0);
            prim->lines[4] = vec2i(4, 5);
            prim->lines[5] = vec2i(5, 6);
            prim->lines[6] = vec2i(6, 7);
            prim->lines[7] = vec2i(7, 4);
            prim->lines[8] = vec2i(0, 4);
            prim->lines[9] = vec2i(1, 5);
            prim->lines[10] = vec2i(2, 6);
            prim->lines[11] = vec2i(3, 7);
        } else if (connType == 3) {
            prim->tris.resize(12);
            // TODO
        } else if (connType == 4) {
            prim->quads.resize(6);
            // TODO
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeVisualAABBPrimitive,
        { /* inputs: */ {
        "boundMin", "boundMax",
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {{"int", "connType", "2"}},
        }, /* category: */ {
        "visualize",
        }});


}
