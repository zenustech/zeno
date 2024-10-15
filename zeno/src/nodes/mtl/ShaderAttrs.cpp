#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <algorithm>

namespace zeno {


struct ShaderInputAttr : ShaderNodeClone<ShaderInputAttr> {
    virtual int determineType(EmissionPass *em) override {
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto attr = get_input2<std::string>("attr");
        auto type = get_input2<std::string>("type");

        if (attr.back() == ')') {
            return em->emitCode(type + "(" + attr + ")");
        } else {
            return em->emitCode(type + "(att_" + attr + ")");
        }
    }
};

ZENDEFNODE(ShaderInputAttr, {
    {
        {"enum pos clr nrm uv tang bitang NoL LoV N T L V H reflectance fresnel instPos instNrm instUv instClr instTang prd.rndf() attrs.localPosLazy() attrs.uniformPosLazy() rayLength isShadowRay worldNrm worldTan worldBTn camFront camUp camRight", "attr", "pos"},
        {"enum float vec2 vec3 vec4 bool", "type", "vec3"},
    },
    {
        {gParamType_Unknown, "out"},
    },
    {},
    {"shader"},
});

struct MakeShaderUniform : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<PrimitiveObject>();
        auto size = get_input2<int>("size");
        prim->resize(size);
        if (has_input("uniformDict")) {
            auto uniformDict = get_input<zeno::DictObject>("uniformDict");
            for (const auto& [key, value] : uniformDict->lut) {
                auto index = std::stoi(key);
                if (auto num = dynamic_cast<const zeno::NumericObject*>(value.get())) {
                    auto value = num->get<zeno::vec3f>();
                    std::vector<vec3f>& attr_arr = prim->add_attr<zeno::vec3f>("pos");
                    if (index < attr_arr.size()) {
                        attr_arr[index] = value;
                    }
                }
                else {
                    throw Exception("Not NumericObject");
                }
            }
        }
        prim->userData().set2("ShaderUniforms", 1);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MakeShaderUniform, {
    {
        {gParamType_Int, "size", "512"},
        {"dict", "uniformDict"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"shader"},
});


struct ShaderUniformAttr : ShaderNodeClone<ShaderUniformAttr> {
    virtual int determineType(EmissionPass *em) override {
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto idx = get_input2<int>("idx");
        auto type = get_input2<std::string>("type");
        return em->emitCode(type + "(vec4(uniforms[" + std::to_string(idx) + "]))");
    }
};

ZENDEFNODE(ShaderUniformAttr, {
                                {
                                    {gParamType_Int, "idx", "0"},
                                    {"enum float vec2 vec3 vec4", "type", "vec3"},
                                },
                                {
                                    {gParamType_Unknown, "out"},
                                },
                                {},
                                {"shader"},
                            });

}
