#include <zeno/zeno.h>
#include <zeno/NumericObject.h>

using namespace zeno;

struct MakeOrthonormalBase : INode {
    virtual void apply() override {
        auto normal = get_input<NumericObject>("normal")->get<vec3f>();
        normal = normalize(normal);
        vec3f tangent, bitangent;
        if (has_input("tangent")) {
            tangent = get_input<NumericObject>("tangent")->get<vec3f>();
            bitangent = cross(normal, tangent);
        } else {
            tangent = vec3f(233, 555, 666);
            bitangent = cross(normal, tangent);
            if (dot(bitangent, bitangent) < 1e-5) {
                tangent = vec3f(-777, -211, -985);
               bitangent = cross(normal, tangent);
            }
        }
        bitangent = normalize(bitangent);
        tangent = cross(bitangent, normal);

        set_output("normal", std::make_shared<NumericObject>(normal));
        set_output("tangent", std::make_shared<NumericObject>(tangent));
        set_output("bitangent", std::make_shared<NumericObject>(bitangent));
    }
};

ZENDEFNODE(MakeOrthonormalBase, {
    {"normal", "tangent"},
    {"normal", "tangent", "bitangent"},
    {},
    {"numeric"},
});
