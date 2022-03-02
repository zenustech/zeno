#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct TreeFinalize : INode {
    virtual void apply() override {
        auto code = EmissionPass{}.finalizeCode({
            "mat_basecolor",
            "mat_metallic",
            "mat_roughness",
            "mat_specular",
            "mat_normal",
            "mat_emission",
            "mat_emitrate",
        }, {
            get_input<IObject>("basecolor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("metallic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("roughness", std::make_shared<NumericObject>(float(0.4f))),
            get_input<IObject>("specular", std::make_shared<NumericObject>(float(0.5f))),
            get_input<IObject>("normal", std::make_shared<NumericObject>(vec3f(0, 0, 1))),
            get_input<IObject>("emission", std::make_shared<NumericObject>(vec3f(0))),
            get_input<IObject>("emitrate", std::make_shared<NumericObject>(float(0.f))),
        });
        set_output2("code", code);
    }
};

ZENDEFNODE(TreeFinalize, {
    {
        {"vec3f", "basecolor", "1,1,1"},
        {"float", "metallic", "0.0"},
        {"float", "roughness", "0.4"},
        {"float", "specular", "0.5"},
        {"vec3f", "normal", "0,0,1"},
        {"vec3f", "emission", "0,0,0"},
        {"float", "emitrate", "0"},
    },
    {
        {"string", "code"},
    },
    {},
    {"tree"},
});


}
