#include "zeno/utils/log.h"
#include "zeno/utils/vec.h"
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <magic_enum.hpp>
#include <algorithm>

namespace zeno {

enum struct SurfaceAttr {
    pos, clr, nrm, uv, tang, bitang, NoL, LoV, N, T, L, V, H, reflectance, fresnel,
    worldNrm, worldTan, worldBTn, 
    camFront, camUp, camRight
};

enum struct InstAttr {
    instIdx, instPos, instNrm, instUv, instClr, instTang
};

enum struct VolumeAttr {};

enum struct RayAttr {
    rayLength, isBackFace, isShadowRay,
};

static std::string shaderAttrDefaultString() {
    auto name = magic_enum::enum_name(SurfaceAttr::pos);
    return std::string(name);
}

static std::string shaderAttrListString() {
    auto list0 = magic_enum::enum_names<SurfaceAttr>();
    auto list1 = magic_enum::enum_names<InstAttr>();
    auto list2 = magic_enum::enum_names<RayAttr>();

    std::string result;

    auto concat = [&](const auto &list) {
        for (auto& ele : list) {
            result += " ";
            result += ele;
        }
    };

    concat(list0); concat(list1); concat(list2);    
   
    result += " prd.rndf() attrs.localPosLazy() attrs.uniformPosLazy()";
    return result;
}

struct ShaderInputAttr : ShaderNodeClone<ShaderInputAttr> {
    virtual int determineType(EmissionPass *em) override {
        auto type = get_input2<std::string>("type");
        return TypeHint.at(type);
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
        {"enum" + shaderAttrListString(), "attr", shaderAttrDefaultString()},
        {"enum " + ShaderDataTypeNamesString, "type", "float"},
    },
    {
        {"shader", "out"},
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
        {"int", "size", "512"},
        {"uniformDict"},
    },
    {
        {"prim"},
    },
    {},
    {"shader"},
});

struct SHParamToUniform : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<PrimitiveObject>();
        if (has_input("SHPrim")) {
            auto prim_in = get_input<zeno::PrimitiveObject>("SHPrim");
            auto& databuffer = prim->add_attr<zeno::vec4f>("buffer");
            size_t sh_verts_count = prim_in->verts.size();
            prim->verts.resize(sh_verts_count*14);
            std::vector<float> & op = prim_in->attr<float>("opacity");
            std::vector<zeno::vec3f> & scale = prim_in->attr<zeno::vec3f>("scale");
            std::vector<zeno::vec3f> & pos = prim_in->attr<zeno::vec3f>("pos");
            std::vector<zeno::vec4f> & rotate = prim_in->attr<zeno::vec4f>("rotate");
        #pragma omp parallel for
            for(auto i=0;i<sh_verts_count;i++){
                for(int j=0;j<12;j++){
                    zeno::vec4f tmp;
                    for(int k=0;k<4;k++){
                        char c_str[10] = "";
                        snprintf(c_str, 10, "SH_%d", j*4 + k);
                        std::vector<float> &data = prim_in->attr<float>(c_str);
                        tmp[k] = data[i];
                    }
                    databuffer[i*14 + j] = tmp;
                }
                databuffer[i*14 + 12] = zeno::vec4f(op[i],pos[i][0],pos[i][1],pos[i][2]);
                databuffer[i*14 + 13] = rotate[i];

            }


            prim->userData().set2("ShaderUniforms", 2);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(SHParamToUniform, {
    {
        {"SHPrim"},
    },
    {
        {"prim"},
    },
    {},
    {"shader"},
});
struct EvalSHColor : ShaderNodeClone<EvalSHColor> {
    virtual int determineType(EmissionPass *em) override {
        em->determineType(get_input("idx").get());
        em->determineType(get_input("dir").get());
        return TypeHint.at("vec3");
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string idx = em->determineExpr(get_input("idx").get());
        std::string dir = em->determineExpr(get_input("dir").get());
        int level = get_input2<int>("SH-Level");
        std::string code=std::string("(") + "GS::EvalSH(uniforms,"+ idx +","+std::to_string(level) +","+ "vec3(params.cam.eye)" + ",attrs.World2ObjectMat"+")"+")";
        printf("Emitcode : %s \n",code.c_str());
        std::string test="vec3(1,0,0)";

        return em->emitCode(code);
    }
};

ZENDEFNODE(EvalSHColor, {
                                {
                                    {"int", "idx", "0"},
                                    {"int", "SH-Level", "0"},
                                    {"vec3", "dir", "0,0,0"}
                                },
                                {
                                    {"vec3", "out"},
                                },
                                {},
                                {"shader"},
                            });

struct EvalGSOpacity : ShaderNodeClone<EvalGSOpacity> {
    virtual int determineType(EmissionPass *em) override {
        em->determineType(get_input("idx").get());
        em->determineType(get_input("pos").get());
        return TypeHint.at("float");
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string idx = em->determineExpr(get_input("idx").get());
        std::string pos = em->determineExpr(get_input("pos").get());
        std::string code=std::string("(float)(") +"GS::EvalGSOpacity("+"uniforms," +idx+","+pos+ ","+"attrs.World2ObjectMat" + "))" ;

        printf("Emitcode : %s \n",code.c_str());

        return em->emitCode(code);
    }
};

ZENDEFNODE(EvalGSOpacity, {
                                {
                                    {"int", "idx", "0"},
                                    {"vec3", "pos", "0,0,0"}
                                },
                                {
                                    {"vec3", "out"},
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
                                    {"int", "idx", "0"},
                                    {"enum float vec2 vec3 vec4", "type", "vec3"},
                                },
                                {
                                    {"shader", "out"},
                                },
                                {},
                                {"shader"},
                            });


}