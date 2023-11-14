#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include <algorithm>
#include "zeno/utils/format.h"

#include <string>
#include "magic_enum.hpp"

namespace zeno
{
struct ShaderTexture2D : ShaderNodeClone<ShaderTexture2D>
{
    virtual int determineType(EmissionPass *em) override {
        auto texId = get_input2<int>("texId");
        auto uvtiling = em->determineType(get_input("uvtiling").get());
        if (has_input("coord")) {
            auto coord = em->determineType(get_input("coord").get());
            if (coord < 2)
                throw zeno::Exception("ShaderTexture2D expect coord to be at least vec2");
        }

        auto type = get_input2<std::string>("type");
        if (type == "float")
            return 1;
        else if (type == "vec2")
            return 2;
        else if (type == "vec3")
            return 3;
        else if (type == "vec4")
            return 4;
        else
            throw zeno::Exception("ShaderTexture2D got bad type: " + type);
    }

    virtual void emitCode(EmissionPass *em) override {
        auto texId = get_input2<int>("texId");
        auto type = get_input2<std::string>("type");
        auto uvtiling = em->determineExpr(get_input("uvtiling").get());
        std::string coord = "att_uv";
        if (has_input("coord")) {
            coord = em->determineExpr(get_input("coord").get());
        }
        em->emitCode(zeno::format("{}(texture2D(zenotex[{}], vec2({}) * {}))", type, texId, coord, uvtiling));
    }
};



struct ShaderTexture3D : ShaderNodeClone<ShaderTexture3D>
{
    enum struct SamplingMethod {
        Closest, Trilinear, Triquadratic, Tricubic, Stochastic
    };

    static std::string methodDefaultString() {
        auto name = magic_enum::enum_name(SamplingMethod::Trilinear);
        return std::string(name);
    }

    static std::string methodListString() {
        auto list = magic_enum::enum_names<SamplingMethod>();

        std::string result;
        for (auto& ele : list) {
            result += " ";
            result += ele;
        }
        return result;
    }

    virtual int determineType(EmissionPass *em) override {
        auto texId = get_input2<int>("texId");
        auto coord = em->determineType(get_input("coord").get());
        if (coord != 3)
            throw zeno::Exception("ShaderTexture3D expect coord to be vec3");

        auto type = get_input2<std::string>("type");

        static const std::map<std::string, int> type_map {
            //{"half", 0},
            //{"float", 1},
            {"vec2", 2},
        };

        if (type_map.count(type) == 0) {
            throw zeno::Exception("ShaderTexture3D got bad type: " + type);
        }

        return type_map.at(type);            
    }

    virtual void emitCode(EmissionPass *em) override {
        auto texId = get_input2<int>("texId");
        auto type = get_input2<std::string>("type");
        auto coord = em->determineExpr(get_input("coord").get());
        
        auto space = get_input2<std::string>("space");
        auto world_space = (space == "World")? "true":"false";

        auto dim = em->determineType(get_input("coord").get());
        auto method = get_input2<std::string>("method");

	    auto casted = magic_enum::enum_cast<SamplingMethod>(method).value_or(SamplingMethod::Trilinear);

        auto order = magic_enum::enum_integer(casted);
        std::string ORDER = std::to_string( order );

        //using DataTypeNVDB0 = float; //nanovdb::Fp32;
        //using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;
        std::string sid = std::to_string(texId);

        std::string DataTypeNVDB = "DataTypeNVDB" + sid;
        std::string GridTypeNVDB = "GridTypeNVDB" + sid;

        em->emitCode(type + "(samplingVDB<"+ ORDER +","+ world_space + "," + DataTypeNVDB +">(vdb_grids[" + sid + "], vec3(" + coord + ")))");
    }
};

ZENDEFNODE(ShaderTexture2D, {
    {
        {"int", "texId", "0"},
        {"coord"},
        {"vec2f", "uvtiling", "1,1"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"shader", "out"},
    },
    {},
    {
        "shader",
    },
});

ZENDEFNODE(ShaderTexture3D, {
    {
        {"int", "texId", "0"},
        {"vec3f", "coord", "0,0,0"},
        {"enum World Local", "space", "World"},
        {"enum vec2", "type", "vec2"},
        {"enum " + ShaderTexture3D::methodListString(), "method", ShaderTexture3D::methodDefaultString()} 
    },
    {
        {"shader", "out"},
    },
    {},
    {
        "shader",
    },
});

} // namespace zeno
