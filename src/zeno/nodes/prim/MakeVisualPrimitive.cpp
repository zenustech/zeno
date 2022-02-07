#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct MakeVisualAABBPrimitive : INode {
    virtual void apply() override {
        auto topless = get_input<NumericObject>("OpenTop")->get<int>();
        auto dx = get_input<NumericObject>("dx")->get<float>();
        auto a = has_input("boundMin")
            ? get_input<NumericObject>("boundMin")->get<vec3f>()
            : vec3f(-0.5, -0.5, -0.5) * dx;
        auto b = has_input("boundMax")
            ? get_input<NumericObject>("boundMax")->get<vec3f>()
            : vec3f(+0.5, +0.5, +0.5) * dx;
        
        auto connType = get_param<std::string>("type");

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

        if (connType == "edges") {
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

        } else if (connType == "trifaces" && topless==0) {
            prim->tris.resize(12);
            prim->tris[0] = vec3i(0, 1, 4);
            prim->tris[1] = vec3i(1, 5, 4);
            prim->tris[2] = vec3i(1, 6, 5);
            prim->tris[3] = vec3i(1, 2, 6);
            prim->tris[4] = vec3i(2, 3, 6);
            prim->tris[5] = vec3i(3, 7, 6);
            prim->tris[6] = vec3i(3, 0, 4);
            prim->tris[7] = vec3i(3, 4, 7);
            prim->tris[8] = vec3i(7, 4, 5);
            prim->tris[9] = vec3i(7, 5, 6);
            prim->tris[10] = vec3i(2, 0, 3);
            prim->tris[11] = vec3i(2, 1, 0);

        }else if (connType == "trifaces" && topless==1) {
            prim->tris.resize(10);
            prim->tris[0] = vec3i(0, 1, 4);
            prim->tris[1] = vec3i(1, 5, 4);
            prim->tris[2] = vec3i(1, 6, 5);
            prim->tris[3] = vec3i(1, 2, 6);
            prim->tris[4] = vec3i(3, 0, 4);
            prim->tris[5] = vec3i(3, 4, 7);
            prim->tris[6] = vec3i(7, 4, 5);
            prim->tris[7] = vec3i(7, 5, 6);
            prim->tris[8] = vec3i(2, 0, 3);
            prim->tris[9] = vec3i(2, 1, 0);

        }
        else if (connType == "quadfaces" && topless == 0) {
            prim->quads.resize(6);
            prim->quads[0] = vec4i(0, 1, 5, 4);
            prim->quads[1] = vec4i(1, 2, 6, 5);
            prim->quads[2] = vec4i(2, 3, 7, 6);
            prim->quads[3] = vec4i(3, 0, 4, 7);
            prim->quads[4] = vec4i(4, 5, 6, 7);
            prim->quads[5] = vec4i(0, 1, 2, 3);
        }else if (connType == "quadfaces" && topless == 1) {
            prim->quads.resize(5);
            prim->quads[0] = vec4i(0, 1, 5, 4);
            prim->quads[1] = vec4i(1, 2, 6, 5);
            prim->quads[2] = vec4i(3, 0, 4, 7);
            prim->quads[3] = vec4i(4, 5, 6, 7);
            prim->quads[4] = vec4i(0, 1, 2, 3);
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeVisualAABBPrimitive,
        { /* inputs: */ {
        {"float", "dx", "1"}, {"vec3f","boundMin","-0.5,-0.5,-0.5"}, {"vec3f","boundMax","0.5,0.5,0.5"}, {"int", "OpenTop", "0"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {{"enum points edges trifaces quadfaces", "type", "edges"}},
        }, /* category: */ {
        "visualize",
        }});
}
