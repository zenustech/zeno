#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/TextureObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/types/UserData.h>

#include <tinygltf/json.hpp>

namespace zeno {

struct ShaderFinalize : INode {
    virtual void apply() override {
        EmissionPass em;

        if (ZImpl(has_input("commonCode")))
            em.commonCode += ZImpl(get_input<StringObject>("commonCode"))->get();

        auto code = em.finalizeCode({
            {1, "mat_base"},
            {3, "mat_basecolor"},
            {1, "mat_roughness"},
            {1, "mat_metallic"},
            {3, "mat_metalColor"},
            {1, "mat_specular"},
            {1, "mat_specularTint"},
            {1, "mat_anisotropic"},
            {1, "mat_anisoRotation"},

            {1, "mat_subsurface"},
            {3, "mat_sssParam"},
            {3, "mat_sssColor"},
            {1, "mat_scatterDistance"},
            {1, "mat_scatterStep"},

            {1, "mat_sheen"},
            {1, "mat_sheenTint"},

            {1, "mat_clearcoat"},
            {3, "mat_clearcoatColor"},
            {1, "mat_clearcoatRoughness"},
            {1, "mat_clearcoatIOR"},

            {1, "mat_specTrans"},
            {3, "mat_transColor"},
            {3, "mat_transTint"},
            {1, "mat_transTintDepth"},
            {1, "mat_transDistance"},
            {3, "mat_transScatterColor"},
            {1, "mat_ior"},

            {1, "mat_diffraction"},
            {3, "mat_diffractColor"},

            {1, "mat_flatness"},
            {1, "mat_shadowReceiver"},
            {1, "mat_shadowTerminatorOffset"},
            {1, "mat_thin"},
            {1, "mat_doubleSide"},
            {3, "mat_normal"},
            {1, "mat_displacement"},
            {1, "mat_smoothness"},
            {1, "mat_emissionIntensity"},
            {3, "mat_emission"},
            {3, "mat_reflectance"}, 
            {1, "mat_opacity"},
            {1, "mat_thickness"},
            {1, "mat_isHair"}

        }, {
            ZImpl(get_input_shader("base", float(1.0f))),
            ZImpl(get_input_shader("basecolor", vec3f(1.0f))),
            ZImpl(get_input_shader("roughness", float(0.4f))),
            ZImpl(get_input_shader("metallic", float(0.0f))),
            ZImpl(get_input_shader("metalColor", vec3f(1.0f))),
            ZImpl(get_input_shader("specular", float(1.0f))),
            ZImpl(get_input_shader("specularTint", float(0.0f))),
            ZImpl(get_input_shader("anisotropic", float(0.0f))),
            ZImpl(get_input_shader("anisoRotation", float(0.0f))),

            ZImpl(get_input_shader("subsurface", float(0.0f))),
            ZImpl(get_input_shader("sssParam", vec3f(1.0f))),
            ZImpl(get_input_shader("sssColor", vec3f(1.0f))),
            ZImpl(get_input_shader("scatterDistance", float(10000))),
            ZImpl(get_input_shader("scatterStep", float(0))),

            ZImpl(get_input_shader("sheen", float(0.0f))),
            ZImpl(get_input_shader("sheenTint", float(0.5f))),

            ZImpl(get_input_shader("clearcoat", float(0.0f))),
            ZImpl(get_input_shader("clearcoatColor", vec3f(1.0f))),
            ZImpl(get_input_shader("clearcoatRoughness", float(0.0f))),
            ZImpl(get_input_shader("clearcoatIOR", float(1.5f))),

            ZImpl(get_input_shader("specTrans", float(0.0f))),
            ZImpl(get_input_shader("transColor", vec3f(1.0f))),
            ZImpl(get_input_shader("transTint", vec3f(1.0f))),
            ZImpl(get_input_shader("transTintDepth", float(10000.0f))),
            ZImpl(get_input_shader("transDistance", float(1.0f))),
            ZImpl(get_input_shader("transScatterColor", vec3f(1.0f))),
            ZImpl(get_input_shader("ior", float(1.5f))),

            ZImpl(get_input_shader("diffraction", float(0.0f))),
            ZImpl(get_input_shader("diffractColor", vec3f(0.0f))),

            ZImpl(get_input_shader("flatness", float(0.0f))),
            ZImpl(get_input_shader("shadowReceiver", float(0.0f))),
            ZImpl(get_input_shader("shadowTerminatorOffset", float(0.0f))),
            ZImpl(get_input_shader("thin", float(0.0f))),
            ZImpl(get_input_shader("doubleSide", float(0.0f))),
            ZImpl(get_input_shader("normal", vec3f(0, 0, 1))),
            ZImpl(get_input_shader("displacement", float(0.0f))),
            ZImpl(get_input_shader("smoothness", float(1.0f))),
            ZImpl(get_input_shader("emissionIntensity", float(1))),
            ZImpl(get_input_shader("emission", vec3f(0))),
            ZImpl(get_input_shader("reflectance", vec3f(1))),
            ZImpl(get_input_shader("opacity", float(0.0))),
            ZImpl(get_input_shader("thickness", float(0.0f))),
            ZImpl(get_input_shader("isHair", float(0.0f)))
        });
        auto commonCode = em.getCommonCode();

        auto sssRadiusMethod = ZImpl(get_input2<std::string>("sssRadius"));
        if (sssRadiusMethod == "Fixed") {
            code += "bool sssFxiedRadius = true;\n";
        } else {
            code += "bool sssFxiedRadius = false;\n";
        }

        vec3f mask_value = (vec3f)ZImpl(get_input2<zeno::vec3i>("mask_value")) / 255.0f;
        code += zeno::format("vec3 mask_value = vec3({}, {}, {});\n", mask_value[0], mask_value[1], mask_value[2]);

        auto mtl = std::make_unique<MaterialObject>();
        mtl->mtlidkey = ZImpl(get_input2<std::string>("mtlid"));
        mtl->frag = std::move(code);

        nlohmann::json j;
        if (ZImpl(has_input("opacity"))) {
            const ShaderData& opa_data = ZImpl(get_input_shader("opacity"));
            if (opa_data.data.index() == 0) {
                std::visit([&](auto&& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                        float opacity = val; // It's actually transparency not opacity
                        opacity = max(0.0f, 1.0f - opacity);
                        j["opacity"] = opacity;
                    }
                    else {
                        throw makeError<UnimplError>("the type of opacity is not int or float");
                    }
                    }, std::get<NumericValue>(opa_data.data));
            }

        }
        mtl->parameters = j.dump();

        if (ZImpl(has_input("extensionsCode")))
            mtl->extensions = ZImpl(get_input<zeno::StringObject>("extensionsCode"))->get();
        {
            if (ZImpl(has_input("tex2dList"))) {
                auto tex2dList = ZImpl(get_input<ListObject>("tex2dList"))->m_impl->get<zeno::Texture2DObject>();
                if (!tex2dList.empty() && !em.tex2Ds.empty()) {
                    throw zeno::makeError("Can not use both way!");
                }
                for (const auto& tex: tex2dList) {
                    em.tex2Ds.push_back(safe_uniqueptr_cast<zeno::Texture2DObject>(tex->clone()));
                }
            }
            if (!em.tex2Ds.empty()) {
                for (const auto& tex: em.tex2Ds) {
                    mtl->tex2Ds.push_back(safe_uniqueptr_cast<zeno::Texture2DObject>(tex->clone()));
                }
                auto texCode = "uniform sampler2D zenotex[32]; \n";
                mtl->common.insert(0, texCode);
            }
        }

        mtl->common = std::move(commonCode);
        set_output("mtl", std::move(mtl));
    }
};

//以下数据类型有可能会接收到其他Shader节点，这时候传过来的是ShaderData，不能走类型判别。
ZENDEFNODE(ShaderFinalize, {
    {
        {gParamType_Float, "base", "1"},
        {gParamType_Vec3f, "basecolor", "1,1,1", Socket_Primitve, ColorVec},
        {gParamType_Float, "roughness", "0.4"},
        {gParamType_Float, "metallic", "0.0"},
        {gParamType_Vec3f, "metalColor","1.0,1.0,1.0", Socket_Primitve, ColorVec},
        {gParamType_Float, "specular", "1.0"},
        {gParamType_Float, "specularTint", "0.0"},
        {gParamType_Float, "anisotropic", "0.0"},
        {gParamType_Float, "anisoRotation", "0.0"},

        {gParamType_Float, "subsurface", "0.0"},
        {"enum Fixed Adaptive", "sssRadius", "Fixed"},
        {gParamType_Vec3f, "sssParam", "1.0,1.0,1.0"},
        {gParamType_Vec3f, "sssColor", "1.0,1.0,1.0", Socket_Primitve, ColorVec},
        {gParamType_Float, "scatterDistance", "10000"},
        {gParamType_Float, "scatterStep", "0"},

        {gParamType_Float, "sheen", "0.0"},
        {gParamType_Float, "sheenTint", "0.0"},

        {gParamType_Float, "clearcoat", "0.0"},
        {gParamType_Vec3f, "clearcoatColor", "1.0,1.0,1.0"},
        {gParamType_Float, "clearcoatRoughness", "0.0"},
        {gParamType_Float, "clearcoatIOR", "1.5"},

        {gParamType_Float, "specTrans", "0.0"},
        {gParamType_Vec3f, "transColor", "1.0,1.0,1.0"},
        {gParamType_Vec3f, "transTint", "1.0,1.0,1.0"},
        {gParamType_Float, "transTintDepth", "10000.0"},
        {gParamType_Float, "transDistance", "10.0"},
        {gParamType_Vec3f, "transScatterColor", "1.0,1.0,1.0"},
        {gParamType_Float, "ior", "1.3"},

        {gParamType_Float, "diffraction", "0.0"},
        {gParamType_Vec3f, "diffractColor", "0.0,0.0,0.0"},

        {gParamType_Float, "flatness", "0.0"},
        {gParamType_Float, "shadowReceiver", "0.0"},
        {gParamType_Float, "shadowTerminatorOffset", "0.0"},
        {gParamType_Float, "thin", "0.0"},
        {gParamType_Float, "doubleSide", "0.0"},
        {gParamType_Vec3f, "normal", "0,0,1"},
        {gParamType_Float, "displacement", "0"},
        {gParamType_Float, "smoothness", "1.0"},
        {gParamType_Float, "emissionIntensity", "1"},
        {gParamType_Vec3f, "emission", "0,0,0"},
        {gParamType_Vec3f, "reflectance", "1,1,1"},
        {gParamType_Float, "opacity", "0"},
        {gParamType_Float, "thickness", "0.0"},
        {gParamType_Float, "isHair", "0.0"},

        {gParamType_String, "commonCode"},
        {gParamType_String, "extensionsCode"},
        {gParamType_String, "mtlid", "Mat1"},
        {gParamType_List, "tex2dList"},//TODO: bate's asset manager
        {gParamType_Vec3i, "mask_value", "0,0,0"},
    },
    {
        {gParamType_Material, "mtl"},
    },
    {
        {"enum CUDA", "backend", "CUDA"},
    },
    {"shader"},
});


}

