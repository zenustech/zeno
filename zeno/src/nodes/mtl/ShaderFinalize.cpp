#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/utils/string.h>
#include <zeno/types/UserData.h>

#include <memory>
#include <string>
#include <iostream>
#include "magic_enum.hpp"

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
            {1, "mat_base"},
            {3, "mat_basecolor"},
            {1, "mat_metallic"},
            {1, "mat_roughness"},
            {1, "mat_specular"},
            {1, "mat_subsurface"},
            {1, "mat_thickness"},
            {3, "mat_sssParam"},
            {3, "mat_sssColor"},
            {1, "mat_specularTint"},
            {1, "mat_anisotropic"},
            {1, "mat_anisoRotation"},
            {1, "mat_sheen"},
            {1, "mat_sheenTint"},
            {1, "mat_clearcoat"},
            {1, "mat_clearcoatGloss"},
            {1, "mat_clearcoatRoughness"},
            {1, "mat_clearcoatIOR"},
            {1, "mat_specTrans"},
            {1, "mat_ior"},
            {1, "mat_flatness"},
            {1, "mat_scatterDistance"},
            {1, "mat_scatterStep"},
            {1, "mat_thin"},
            {1, "mat_doubleSide"},
            {3, "mat_normal"},
            {1, "mat_displacement"},
            {1, "mat_smoothness"},
            {1, "mat_emissionIntensity"},
            {3, "mat_emission"},
            {3, "mat_reflectance"}, 
            {1, "mat_opacity"},

            {1, "vol_depth"},
            {1, "vol_extinction"},
            {3, "vol_sample_albedo"},
            {1, "vol_sample_anisotropy"},

            {1, "vol_sample_density"},
            {3, "vol_sample_emission"},

        }, {
            get_input<IObject>("base", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("basecolor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("metallic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("roughness", std::make_shared<NumericObject>(float(0.4f))),
            get_input<IObject>("specular", std::make_shared<NumericObject>(float(0.5f))),
            get_input<IObject>("subsurface", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("thickness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sssParam", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("sssColor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("specularTint", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("anisotropic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("anisoRotation", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sheen", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("sheenTint", std::make_shared<NumericObject>(float(0.5f))),
            get_input<IObject>("clearcoat", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("clearcoatGloss", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("clearcoatRoughness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("clearcoatIOR", std::make_shared<NumericObject>(float(1.5f))),
            get_input<IObject>("specTrans", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("ior", std::make_shared<NumericObject>(float(1.5f))),
            get_input<IObject>("flatness", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("scatterDistance", std::make_shared<NumericObject>(float(10000))),
            get_input<IObject>("scatterStep", std::make_shared<NumericObject>(float(0))),
            get_input<IObject>("thin", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("doubleSide", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("normal", std::make_shared<NumericObject>(vec3f(0, 0, 1))),
            get_input<IObject>("displacement", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("smoothness", std::make_shared<NumericObject>(float(1.0f))),
            get_input<IObject>("emissionIntensity", std::make_shared<NumericObject>(float(1))),
            get_input<IObject>("emission", std::make_shared<NumericObject>(vec3f(0))),
            get_input<IObject>("reflectance", std::make_shared<NumericObject>(vec3f(1))),
            get_input<IObject>("opacity", std::make_shared<NumericObject>(float(0.0))),
            
            get_input<IObject>("vol_depth", std::make_shared<NumericObject>((float)(99))),
            get_input<IObject>("vol_extinction", std::make_shared<NumericObject>(float(1))),
            get_input<IObject>("vol_sample_albedo", std::make_shared<NumericObject>(vec3f(0.5))),
            get_input<IObject>("vol_sample_anisotropy", std::make_shared<NumericObject>(float(0))),

            get_input<IObject>("vol_sample_density", std::make_shared<NumericObject>(float(0))),
            get_input<IObject>("vol_sample_emission", std::make_shared<NumericObject>(vec3f(0))),
            
        });
        auto commonCode = em.getCommonCode();

        auto mtl = std::make_shared<MaterialObject>();
        mtl->frag = std::move(code);

        if (has_input("extensionsCode"))
            mtl->extensions = get_input<zeno::StringObject>("extensionsCode")->get();

        if (has_input("tex2dList"))
        {
            auto tex2dList = get_input<ListObject>("tex2dList")->get<zeno::Texture2DObject>();
            for (const auto tex: tex2dList)
            {
                auto texId = mtl->tex2Ds.size();
			    mtl->tex2Ds.push_back(tex);
            }

            auto texCode = "uniform sampler2D zenotex[32]; \n";
            mtl->common.insert(0, texCode);
        }

        if (has_input("tex3dList"))
        {
            int   vol_depth = (int)get_input2<float>("vol_depth");
            float vol_extinction = get_input2<float>("vol_extinction");

            vol_depth = clamp(vol_depth, 9, 99);

            commonCode += "static const int   _vol_depth = " + std::to_string(vol_depth) + ";\n";
            commonCode += "static const float _vol_extinction = " + std::to_string(vol_extinction) + ";\n";

            auto tex3dList = get_input<ListObject>("tex3dList")->getRaw(); //get<zeno::StringObject>();

            for (const auto& tex3d : tex3dList) {

                const auto ele = dynamic_cast<zeno::StringObject*>(tex3d);
                if (ele == nullptr) {
                    auto texObject = std::dynamic_pointer_cast<zeno::TextureObjectVDB>(tex3d->clone());
                    mtl->tex3Ds.push_back(texObject); 
                    continue;
                }

                auto path = ele->get();
                auto ud = ele->userData();

                const std::string _key_ = "channel";
                std::string channel_string = "0";

                if (ud.has(_key_)) {

                    if (ud.isa<zeno::StringObject>(_key_)) {
                        //auto get = ud.get<zeno::StringObject>("channel");
                        channel_string = ud.get2<std::string>(_key_);

                    } else if (ud.isa<zeno::NumericObject>(_key_)) {
                        auto channel_number = ud.get2<int>(_key_);
                        channel_number = max(0, channel_number);
                        channel_string = std::to_string(channel_number);
                    } 
                }

                auto toVDB = std::make_shared<TextureObjectVDB>();
                toVDB->path = path;
                toVDB->channel = channel_string;
                toVDB->eleType = TextureObjectVDB::ElementType::Fp32;

                mtl->tex3Ds.push_back(std::move(toVDB)); 
            }

            std::stringstream type_string;

            // using DataTypeNVDB = float; nanovdb::Fp32;
            // using GridTypeNVDB = nanovdb::NanoGrid<DataTypeNVDB>;

            for (size_t i=0; i<mtl->tex3Ds.size(); ++i) {
                auto& tex3d = mtl->tex3Ds.at(i);
                auto idx = std::to_string(i);

                auto nano_type = magic_enum::enum_name(tex3d->eleType);

                type_string << "using DataTypeNVDB" << idx << " = nanovdb::" << nano_type << "; \n";
                type_string << "using GridTypeNVDB" << idx << " = nanovdb::NanoGrid<DataTypeNVDB" << idx + ">; \n";
            }
            commonCode += type_string.str();
            //std::cout << commonCode << std::endl; 
            auto ud = get_input<ListObject>("tex3dList")->userData();

            if ( ud.has("transform") ) {
            
                auto transformString = ud.get2<std::string>("transform");
                mtl->transform = transformString;
            }
        }

        mtl->common = std::move(commonCode);
        //if (has_input("mtlid"))
        //{
            mtl->mtlidkey = get_input2<std::string>("mtlid");
        //}

        set_output("mtl", std::move(mtl));
    }
};

ZENDEFNODE(ShaderFinalize, {
    {
        {"float", "base", "1"},
        {"vec3f", "basecolor", "1,1,1"},
        {"float", "metallic", "0.0"},
        {"float", "roughness", "0.4"},
        {"float", "specular", "0.5"},
        {"float", "subsurface", "0.0"},
        {"float", "thickness", "0.0"},
        {"vec3f", "sssParam", "1,1,1"},
        {"vec3f", "sssColor", "1.0,1.0,1.0"},
        {"float", "specularTint", "0.0"},
        {"float", "anisotropic", "0.0"},
        {"float", "anisoRotation", "0.0"},
        {"float", "sheen", "0.0"},
        {"float", "sheenTint", "0.0"},
        {"float", "clearcoat", "0.0"},
        {"float", "clearcoatGloss", "1.0"},
        {"float", "clearcoatRoughness", "0.0"},
        {"float", "clearcoatIOR", "1.5"},
        {"float", "specTrans", "0.0"},
        {"float", "ior", "1.5"},
        {"float", "flatness", "0.0"},
        {"float", "scatterDistance", "10000"},
        {"float", "scatterStep", "0"},
        {"float", "thin", "0.0"},
        {"float", "doubleSide", "0.0"},
        {"vec3f", "normal", "0,0,1"},
        {"float", "displacement", "0"},
        {"float", "smoothness", "1.0"},
        {"float", "emissionIntensity", "1"},
        {"vec3f", "emission", "0,0,0"},
        {"vec3f", "reflectance", "1,1,1"},
        {"float", "opacity", "0"},
        {"string", "commonCode"},
        {"string", "extensionsCode"},
        {"string", "mtlid", "Mat1"},
        {"list", "tex2dList"},//TODO: bate's asset manager
        {"list", "tex3dList"},

        {"float", "vol_depth",     "99"},
        {"float", "vol_extinction", "1"},
        {"vec3f", "vol_sample_albedo", "0.5,0.5,0.5"},

        {"float", "vol_sample_anisotropy", "0"},

        {"float", "vol_sample_density", "0"},
        {"vec3f", "vol_sample_emission", "0,0,0"}
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
