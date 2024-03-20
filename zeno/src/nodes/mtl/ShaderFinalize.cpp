#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/types/UserData.h>

namespace zeno {

struct ShaderFinalize : INode {
    virtual void apply() override {
        EmissionPass em;

        if (has_input("commonCode"))
            em.commonCode += get_input<StringObject>("commonCode")->get();

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
            get_input<IObject>("base", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("basecolor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("roughness", std::make_shared<NumericObject>(float(0.4f))),
            get_input<IObject>("metallic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("metalColor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("specular", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("specularTint", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("anisotropic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("anisoRotation", std::make_shared<NumericObject>(float(0.0f))),

            get_input<IObject>("subsurface", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sssParam", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("sssColor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("scatterDistance", std::make_shared<NumericObject>(float(10000))),
            get_input<IObject>("scatterStep", std::make_shared<NumericObject>(float(0))),

            get_input<IObject>("sheen", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sheenTint", std::make_shared<NumericObject>(float(0.5f))),

            get_input<IObject>("clearcoat", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("clearcoatColor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("clearcoatRoughness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("clearcoatIOR", std::make_shared<NumericObject>(float(1.5f))),

            get_input<IObject>("specTrans", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("transColor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("transTint", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("transTintDepth", std::make_shared<NumericObject>(float(10000.0f))),
            get_input<IObject>("transDistance", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("transScatterColor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("ior", std::make_shared<NumericObject>(float(1.5f))),

            get_input<IObject>("diffraction", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("diffractColor", std::make_shared<NumericObject>(vec3f(0.0f))),

            get_input<IObject>("flatness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("shadowReceiver", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("thin", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("doubleSide", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("normal", std::make_shared<NumericObject>(vec3f(0, 0, 1))),
            get_input<IObject>("displacement", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("smoothness", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("emissionIntensity", std::make_shared<NumericObject>(float(1))),
            get_input<IObject>("emission", std::make_shared<NumericObject>(vec3f(0))),
            get_input<IObject>("reflectance", std::make_shared<NumericObject>(vec3f(1))),
            get_input<IObject>("opacity", std::make_shared<NumericObject>(float(0.0))),
            get_input<IObject>("thickness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("isHair", std::make_shared<NumericObject>(float(0.0f)))
        });
        auto commonCode = em.getCommonCode();

        auto sssRadiusMethod = get_input2<std::string>("sssRadius");
        if (sssRadiusMethod == "Fixed") {
            code += "bool sssFxiedRadius = true;\n";
        } else {
            code += "bool sssFxiedRadius = false;\n";
        }

        vec3f mask_value = (vec3f)get_input2<vec3i>("mask_value") / 255.0f;
        code += zeno::format("vec3 mask_value = vec3({}, {}, {});\n", mask_value[0], mask_value[1], mask_value[2]);

        auto mtl = std::make_shared<MaterialObject>();
        mtl->mtlidkey = get_input2<std::string>("mtlid");
        mtl->frag = std::move(code);

        if (has_input("extensionsCode"))
            mtl->extensions = get_input<zeno::StringObject>("extensionsCode")->get();

        {
            if (has_input("tex2dList")) {
                auto tex2dList = get_input<ListObject>("tex2dList")->get<zeno::Texture2DObject>();
                if (!tex2dList.empty() && !em.tex2Ds.empty()) {
                    throw zeno::makeError("Can not use both way!");
                }
                for (const auto& tex: tex2dList) {
                    em.tex2Ds.push_back(tex);
                }
            }
            if (!em.tex2Ds.empty()) {
                for (const auto& tex: em.tex2Ds) {
                    mtl->tex2Ds.push_back(tex);
                }
                auto texCode = "uniform sampler2D zenotex[32]; \n";
                mtl->common.insert(0, texCode);
            }
        }

        mtl->common = std::move(commonCode);
        set_output("mtl", std::move(mtl));
    }
};

ZENDEFNODE(ShaderFinalize, {
    {
        {"float", "base", "1"},
        {"colorvec3f", "basecolor", "1,1,1"},
        {"float", "roughness", "0.4"},
        {"float", "metallic", "0.0"},
        {"colorvec3f", "metalColor","1.0,1.0,1.0"},
        {"float", "specular", "1.0"},
        {"float", "specularTint", "0.0"},
        {"float", "anisotropic", "0.0"},
        {"float", "anisoRotation", "0.0"},

        {"float", "subsurface", "0.0"},
        {"enum Fixed Adaptive", "sssRadius", "Fixed"},
        {"vec3f", "sssParam", "1.0,1.0,1.0"},
        {"colorvec3f", "sssColor", "1.0,1.0,1.0"},
        {"float", "scatterDistance", "10000"},
        {"float", "scatterStep", "0"},

        {"float", "sheen", "0.0"},
        {"float", "sheenTint", "0.0"},

        {"float", "clearcoat", "0.0"},
        {"vec3f", "clearcoatColor", "1.0,1.0,1.0"},
        {"float", "clearcoatRoughness", "0.0"},
        {"float", "clearcoatIOR", "1.5"},

        {"float", "specTrans", "0.0"},
        {"vec3f", "transColor", "1.0,1.0,1.0"},
        {"vec3f", "transTint", "1.0,1.0,1.0"},
        {"float", "transTintDepth", "10000.0"},
        {"float", "transDistance", "10.0"},
        {"vec3f", "transScatterColor", "1.0,1.0,1.0"},
        {"float", "ior", "1.3"},

        {"float", "diffraction", "0.0"},
        {"vec3f", "diffractColor", "0.0,0.0,0.0"},

        {"float", "flatness", "0.0"},
        {"float", "shadowReceiver", "0.0"},
        {"float", "thin", "0.0"},
        {"float", "doubleSide", "0.0"},
        {"vec3f", "normal", "0,0,1"},
        {"float", "displacement", "0"},
        {"float", "smoothness", "1.0"},
        {"float", "emissionIntensity", "1"},
        {"vec3f", "emission", "0,0,0"},
        {"vec3f", "reflectance", "1,1,1"},
        {"float", "opacity", "0"},
        {"float", "thickness", "0.0"},
        {"float", "isHair", "0.0"},

        {"string", "commonCode"},
        {"string", "extensionsCode"},
        {"string", "mtlid", "Mat1"},
        {"list", "tex2dList"},//TODO: bate's asset manager
        {"vec3i", "mask_value", "0,0,0"},
    },
    {
        {"MaterialObject", "mtl"},
    },
    {
        {"enum CUDA", "backend", "CUDA"},
    },
    {"shader"},
});


}
