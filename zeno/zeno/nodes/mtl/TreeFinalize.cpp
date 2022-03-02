#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct TreeFinalize : INode {
    virtual void apply() override {
        EmissionPass em;
        auto backend = get_param<std::string>("backend");
        if (backend == "HLSL")
            em.backend = em.HLSL;
        else if (backend == "GLSL")
            em.backend = em.GLSL;

        auto code = em.finalizeCode({
            {3, "mat_basecolor"},
            {1, "mat_metallic"},
            {1, "mat_roughness"},
            {1, "mat_specular"},
            {3, "mat_normal"},
            {3, "mat_emission"},
            {1, "mat_emitrate"},
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
    {
        {"enum GLSL HLSL", "backend", "GLSL"},
    },
    {"tree"},
});


}
