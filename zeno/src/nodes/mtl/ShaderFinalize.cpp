#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct ShaderFinalize : INode {
    virtual void apply() override {
        EmissionPass em;
        auto backend = get_param<std::string>("backend");
        if (backend == "HLSL")
            em.backend = em.HLSL;
        else if (backend == "GLSL")
            em.backend = em.GLSL;

        if (has_input("commonCode"))
            em.commonCode += get_input<StringObject>("commonCode")->get();

        auto code = em.finalizeCode({
            {3, "mat_basecolor"},
            {1, "mat_metallic"},
            {1, "mat_roughness"},
            {1, "mat_specular"},
            {1, "mat_subsurface"},
            {1, "mat_thickness"},
            {3, "mat_sssParam"},
            {3, "mat_sssColor"},
            {1, "mat_foliage"},
            {1, "mat_skin"},
            {1, "mat_curvature"},
            {1, "mat_specularTint"},
            {1, "mat_anisotropic"},
            {1, "mat_sheen"},
            {1, "mat_sheenTint"},
            {1, "mat_clearcoat"},
            {1, "mat_clearcoatGloss"},
            {3, "mat_normal"},
            {3, "mat_emission"},
            {1, "mat_zenxposure"},
            {1, "mat_ao"},
            {1, "mat_toon"},
            {1, "mat_stroke"},
            {3, "mat_shape"},
            {1, "mat_style"},
            {1, "mat_strokeNoise"},
            {3,"mat_shad"},
            {3,"mat_strokeTint"},
            {1,"mat_opacity"},
            {1,"mat_reflection"},
            {1,"mat_reflectID"},
            {1,"mat_isCamera"},
            {1,"mat_isVoxelDomain"}
        }, {
            get_input<IObject>("basecolor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("metallic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("roughness", std::make_shared<NumericObject>(float(0.4f))),
            get_input<IObject>("specular", std::make_shared<NumericObject>(float(0.5f))),
            get_input<IObject>("subsurface", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("thickness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sssParam", std::make_shared<NumericObject>(vec3f(0.0f))),
            get_input<IObject>("sssColor", std::make_shared<NumericObject>(vec3f(0.0f))),
            get_input<IObject>("foliage", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("skin", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("curvature", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("specularTint", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("anisotropic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sheen", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sheenTint", std::make_shared<NumericObject>(float(0.5f))),
            get_input<IObject>("clearcoat", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("clearcoatGloss", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("normal", std::make_shared<NumericObject>(vec3f(0, 0, 1))),
            get_input<IObject>("emission", std::make_shared<NumericObject>(vec3f(0))),
            get_input<IObject>("exposure", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("ao", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("toon", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("stroke", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("shape", std::make_shared<NumericObject>(vec3f(-0.5,0.5,0))),
            get_input<IObject>("style", std::make_shared<NumericObject>(float(1.0))),
            get_input<IObject>("strokeNoise", std::make_shared<NumericObject>(float(1))),
            get_input<IObject>("shad", std::make_shared<NumericObject>(vec3f(0,0,0))),
            get_input<IObject>("strokeTint", std::make_shared<NumericObject>(vec3f(0,0,0))),
            get_input<IObject>("opacity", std::make_shared<NumericObject>(float(0.0))),
            get_input<IObject>("reflection", std::make_shared<NumericObject>(float(0.0))),
            get_input<IObject>("reflectID", std::make_shared<NumericObject>(float(-1))),
            get_input<IObject>("isCamera", std::make_shared<NumericObject>(float(0))),
            get_input<IObject>("isVoxelDomain", std::make_shared<NumericObject>(float(0))),

            
        });
        auto commonCode = em.getCommonCode();

        auto mtl = std::make_shared<MaterialObject>();
        mtl->frag = std::move(code);
        mtl->common = std::move(commonCode);
        if (has_input("extensionsCode"))
            mtl->extensions = get_input<zeno::StringObject>("extensionsCode")->get();

        if (has_input("tex2dList"))
        {
            auto tex2dList = get_input<ListObject>("tex2dList")->get<zeno::Texture2DObject>();
            for (const auto &tex: tex2dList)
            {
                auto texId = mtl->tex2Ds.size();
                auto texCode = "uniform sampler2D zenotex" + std::to_string(texId) + ";\n";
			    mtl->tex2Ds.push_back(tex);
                mtl->common.insert(0, texCode);
            }
        }

        //if (has_input("mtlid"))
        //{
            mtl->mtlidkey = get_input2<std::string>("mtlid");
        //}

        set_output("mtl", std::move(mtl));
    }
};

ZENDEFNODE(ShaderFinalize, {
    {
        {"vec3f", "basecolor", "1,1,1"},
        {"float", "metallic", "0.0"},
        {"float", "roughness", "0.4"},
        {"float", "specular", "0.5"},
        {"float", "subsurface", "0.0"},
        {"float", "thickness", "0.0"},
        {"vec3f", "sssParam", "0,0,0"},
        {"vec3f", "sssColor", "0,0,0"},
        {"float", "foliage", "0"},
        {"float", "skin", "0"},
        {"float", "curvature", "0"},
        {"float", "specularTint", "0.0"},
        {"float", "anisotropic", "0.0"},
        {"float", "sheen", "0.0"},
        {"float", "sheenTint", "0.0"},
        {"float", "clearcoat", "0.0"},
        {"float", "clearcoatGloss", "1.0"},
        {"vec3f", "normal", "0,0,1"},
        {"vec3f", "emission", "0,0,0"},
        {"float", "exposure", "1.0"},
        {"float", "ao", "1.0"},
        {"float", "toon", "0.0"},
        {"float", "stroke", "1.0"},
        {"vec3f", "shape", "-0.5,0.5,0"},
        {"float", "style", "1.0"},
        {"float", "strokeNoise", "1"},
        {"vec3f", "shad", "0,0,0"},
        {"vec3f", "strokeTint", "0,0,0"},
        {"float", "opacity", "0"},
        {"float", "reflection", "0"},
        {"float", "reflectID", "-1"},
        {"float", "isCamera", "0"},
        {"float", "isVoxelDomain", "0"},
        {"string", "commonCode"},
        {"string", "extensionsCode"},
        {"string", "mtlid", "Mat1"},
        {"list", "tex2dList"},//TODO: bate's asset manager
    },
    {
        {"MaterialObject", "mtl"},
    },
    {
        {"enum GLSL HLSL", "backend", "GLSL"},
    },
    {"shader"},
});


}
