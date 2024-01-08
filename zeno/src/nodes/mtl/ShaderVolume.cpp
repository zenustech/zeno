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

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

namespace zeno {

struct ShaderVolume : INode {

virtual void apply() override {
        EmissionPass em;

        if (has_input("commonCode")) {
            em.commonCode += get_input<StringObject>("commonCode")->get();
        }

        auto code = em.finalizeCode({

            {1, "depth"},
            {1, "extinction"},
            {3, "albedo"},
            {1, "anisotropy"},

            {1, "density"},
            {3, "emission"},

        }, {
           
            get_input<IObject>("depth", std::make_shared<NumericObject>((float)(999))),
            get_input<IObject>("extinction", std::make_shared<NumericObject>(float(1))),
            get_input<IObject>("albedo", std::make_shared<NumericObject>(vec3f(0.5))),
            get_input<IObject>("anisotropy", std::make_shared<NumericObject>(float(0))),

            get_input<IObject>("density", std::make_shared<NumericObject>(float(0))),
            get_input<IObject>("emission", std::make_shared<NumericObject>(vec3f(0))),
            
        });
        
        auto mtl = std::make_shared<MaterialObject>();
        mtl->frag = std::move(code);

        if (has_input("extensionsCode")) {
            mtl->extensions = get_input<zeno::StringObject>("extensionsCode")->get();
        }

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
            }

        int vol_depth = (int)get_input2<float>("depth");
        float vol_extinction = get_input2<float>("extinction");

        auto EmissionScale = get_input2<std::string>("EmissionScale:");
        em.commonCode += "#define VolumeEmissionScale VolumeEmissionScaleType::" + EmissionScale + "\n";

        vol_depth = clamp(vol_depth, 9, 9999);
        vol_extinction = clamp(vol_extinction, 1e-5, 1e+5);

        std::string parameters = "";
        {
            using namespace rapidjson;
            Document d; d.SetObject();
            auto& allocator = d.GetAllocator();

            Value s = Value();
            s.SetInt(vol_depth);
            d.AddMember("vol_depth", s, allocator);

            s = Value();
            s.SetFloat(vol_extinction); 
            d.AddMember("vol_extinction", s, allocator);

            StringBuffer buffer;
            Writer<StringBuffer> writer(buffer);
            d.Accept(writer);
            parameters = buffer.GetString();
        }
        mtl->parameters = parameters;
        mtl->mtlidkey = get_input2<std::string>("mtlid");

        if (has_input("tex3dList"))
        {
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
            em.commonCode += type_string.str();
            //std::cout << commonCode << std::endl; 
            auto ud = get_input<ListObject>("tex3dList")->userData();

            if ( ud.has("transform") ) {
            
                auto transformString = ud.get2<std::string>("transform");
                mtl->transform = transformString;
            }
        }

        mtl->common = std::move(em.commonCode);
        set_output("mtl", std::move(mtl));
    }
};

ZENDEFNODE(ShaderVolume, {
    {
        {"string", "mtlid", "VolMat1"},
        {"string", "commonCode"},
        {"string", "extensionsCode"},

        {"list", "tex2dList"},
        {"list", "tex3dList"},

        {"float", "depth",    "999"},
        {"float", "extinction", "1"},
        {"float", "anisotropy", "0"},

        {"vec3f", "albedo", "0.5,0.5,0.5"},
        {"float", "density", "0"},
        {"vec3f", "emission", "0.0,0.0,0.0"}
    },
    { {"MaterialObject", "mtl"} },
    {
        {"enum RatioTracking", "Transmittance", "RatioTracking"},
        {"enum Raw Density Absorption", "EmissionScale", "Raw"},
    },
    {"shader"}
});

}