//
// Created by AS on 2023/2/13.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <unordered_set>

#include <random>
#include <algorithm>

namespace zeno {
struct PrimSuperFormula : zeno::INode {

    virtual void apply() override {
        auto outprim = std::make_shared<zeno::PrimitiveObject>();
        auto hasLines = get_input2<bool>("hasLines");
        auto close = get_input2<bool>("close");

        int segments = get_input2<int>("segments");
        float scale = get_input2<float>("scale");
        float a = get_input2<float>("a");
        float b = get_input2<float>("b");
        float m = get_input2<float>("m");
        float n1 = get_input2<float>("n1");
        float n2 = get_input2<float>("n2");
        float n3 = get_input2<float>("n3");

        float tau = 6.28318530718;
        float step = tau / (float)segments;
        float phi = 0;

        for (size_t i = 0; i < segments; ++i) {
            float mp4 = (m * phi) / 4.0;
            float term_a = pow((abs(cos(mp4) / a)), n2);
            float term_b = pow((abs(sin(mp4) / b)), n3);
            float r = pow((term_a + term_b), (-1 / n1));
            float x = cos(phi) * scale * r;
            float y = sin(phi) * scale * r;
            zeno::vec3f pos = zeno::vec3f(x, y, 0.0);

            outprim->verts.push_back(pos);
            phi += step;
        }

        auto line_num = (intptr_t)(outprim->verts.size() - 1);
        if (hasLines) {
            for (intptr_t l = 0; l < line_num; ++l) {
                outprim->lines.emplace_back(l, l + 1);
            }
            if (close) {
                outprim->lines.emplace_back(0, line_num);
            }
            outprim->lines.update();
        }

        set_output("output", std::move(outprim));
    }
};
ZENDEFNODE(PrimSuperFormula, {{
                                  /* inputs: */
                                  {gParamType_Int, "segments", "1000"},
                                  {gParamType_Float, "scale", "1.0"},

                                  {gParamType_Float, "a", "1.0"},
                                  {gParamType_Float, "b", "1.0"},
                                  {gParamType_Float, "m", "16.0"},
                                  {gParamType_Float, "n1", "-8.0"},
                                  {gParamType_Float, "n2", "12.0"},
                                  {gParamType_Float, "n3", "10.0"},

                                  {gParamType_Bool, "hasLines", "1"},
                                  {gParamType_Bool, "close", "1"},
                              },

                              {
                                  /* outputs: */
                                  {gParamType_Primitive, "output"},
                              },

                              {
                                  /* params: */

                              },

                              {
                                  /* category: */
                                  "primitive",
                              }});
}