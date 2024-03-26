#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/TextureObject.h>
#include "zeno/types/HeatmapObject.h"
#include <zeno/utils/string.h>
#include <algorithm>
#include "zeno/utils/format.h"
#include "zeno/utils/logger.h"
#include <tinygltf/stb_image_write.h>
#include <filesystem>

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

        auto cihou = get_input2<bool>("cihou");
        
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

        em->emitCode(type + "(samplingVDB<"+ ORDER +","+ world_space + "," + DataTypeNVDB +">(vdb_grids[" + sid + "], vec3(" + coord + "), attrs, " + std::to_string(cihou) + "))");
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
        {"bool", "cihou", "0"},
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

struct SmartTexture2D : ShaderNodeClone<SmartTexture2D>
{
    static constexpr char
        texWrapping[] = "REPEAT MIRRORED_REPEAT CLAMP_TO_EDGE CLAMP_TO_BORDER",
        texFiltering[] = "NEAREST LINEAR NEAREST_MIPMAP_NEAREST LINEAR_MIPMAP_NEAREST NEAREST_MIPMAP_LINEAR LINEAR_MIPMAP_LINEAR";
    virtual int determineType(EmissionPass *em) override {
        auto uvtiling = em->determineType(get_input("uvtiling").get());
        if (has_input("coord")) {
            auto coord = em->determineType(get_input("coord").get());
            if (coord < 2)
                throw zeno::Exception("ShaderTexture2D expect coord to be at least vec2");
        }

        auto type = get_input2<std::string>("type");
        if (type == "float" || type == "R" || type == "G" || type == "B" || type == "A")
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
        auto texId = em->tex2Ds.size();
        auto tex = std::make_shared<zeno::Texture2DObject>();
        auto texture_path = get_input2<std::string>("path");
        if(!std::filesystem::exists(std::filesystem::u8path(texture_path))){
            zeno::log_warn("texture file not found!");
            auto type = get_input2<std::string>("type");
            vec4f number= vec4f(0,0,0,0);
            if(has_input2<float>("value"))
            {
              number[0] = get_input2<float>("value");
            }
            if(has_input2<vec2f>("value"))
            {
              auto in = get_input2<vec2f>("value");
              number[0] = in[0];
              number[1] = in[1];
            }
            if(has_input2<vec3f>("value"))
            {
              auto in = get_input2<vec3f>("value");
              number[0] = in[0];
              number[1] = in[1];
              number[2] = in[2];
            }
            if(has_input2<vec4f>("value"))
            {
              auto in = get_input2<vec4f>("value");
              number[0] = in[0];
              number[1] = in[1];
              number[2] = in[2];
              number[3] = in[3];
            }

            if (type == "float" || type == "R")
                em->emitCode(zeno::format("{}",number[0]));
            else if (type == "G")
                em->emitCode(zeno::format("{}",number[1]));
            else if (type == "B")
                em->emitCode(zeno::format("{}",number[2]));
            else if (type == "A")
                em->emitCode(zeno::format("{}",number[3]));
            else if (type == "vec2")
                em->emitCode(zeno::format("vec2({},{})",number[0],number[1]));
            else if (type == "vec3")
                em->emitCode(zeno::format("vec3({},{},{})",number[0],number[1],number[2]));
            else if (type == "vec4")
                em->emitCode(zeno::format("vec4({},{},{},{})",number[0],number[1],number[2],number[3]));
            else
                throw zeno::Exception("ShaderTexture2D got bad type: " + type);
            
            return;
            
        }

        tex->path = texture_path;
        if (has_input("heatmap")) {
            if (tex->path.empty()) {
                std::srand(std::time(0));
                tex->path = std::filesystem::temp_directory_path().string() + '/' + "heatmap-" + std::to_string(std::rand()) + ".png";
            }
            auto heatmap = get_input<zeno::HeatmapObject>("heatmap");
            std::vector<uint8_t> col;
            int width = heatmap->colors.size();
            int height = width;
            col.reserve(width * height * 3);
            for (auto i = 0; i < height; i++) {
                for (auto & color : heatmap->colors) {
                    col.push_back(zeno::clamp(int(color[0] * 255.99), 0, 255));
                    col.push_back(zeno::clamp(int(color[1] * 255.99), 0, 255));
                    col.push_back(zeno::clamp(int(color[2] * 255.99), 0, 255));
                }
            }
            stbi_flip_vertically_on_write(false);
            stbi_write_png(tex->path.c_str(), width, height, 3, col.data(), 0);
        }

    #define SET_TEX_WRAP(TEX, WRAP)                                    \
        if (WRAP == "REPEAT")                                          \
            TEX->WRAP = Texture2DObject::TexWrapEnum::REPEAT;          \
        else if (WRAP == "MIRRORED_REPEAT")                            \
            TEX->WRAP = Texture2DObject::TexWrapEnum::MIRRORED_REPEAT; \
        else if (WRAP == "CLAMP_TO_EDGE")                              \
            TEX->WRAP = Texture2DObject::TexWrapEnum::CLAMP_TO_EDGE;   \
        else if (WRAP == "CLAMP_TO_BORDER")                            \
            TEX->WRAP = Texture2DObject::TexWrapEnum::CLAMP_TO_BORDER; \
        else                                                           \
            throw zeno::Exception(#WRAP + WRAP);

        auto wrapS = get_input<zeno::StringObject>("wrapT")->get();
        SET_TEX_WRAP(tex, wrapS)
        auto wrapT = get_input<zeno::StringObject>("wrapS")->get();
        SET_TEX_WRAP(tex, wrapT)

    #undef SET_TEX_WRAP

    #define SET_TEX_FILTER(TEX, FILTER)                                           \
        if (FILTER == "NEAREST")                                                  \
            TEX->FILTER = Texture2DObject::TexFilterEnum::NEAREST;                \
        else if (FILTER == "LINEAR")                                              \
            TEX->FILTER = Texture2DObject::TexFilterEnum::LINEAR;                 \
        else if (FILTER == "NEAREST_MIPMAP_NEAREST")                              \
            TEX->FILTER = Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_NEAREST; \
        else if (FILTER == "LINEAR_MIPMAP_NEAREST")                               \
            TEX->FILTER = Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_NEAREST;  \
        else if (FILTER == "NEAREST_MIPMAP_LINEAR")                               \
            TEX->FILTER = Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_LINEAR;  \
        else if (FILTER == "LINEAR_MIPMAP_LINEAR")                                \
            TEX->FILTER = Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_LINEAR;   \
        else                                                                      \
            throw zeno::Exception(#FILTER + FILTER);

        auto minFilter = get_input<zeno::StringObject>("minFilter")->get();
        SET_TEX_FILTER(tex, minFilter)
        auto magFilter = get_input<zeno::StringObject>("magFilter")->get();
        SET_TEX_FILTER(tex, magFilter)

    #undef SET_TEX_FILTER
        em->tex2Ds.push_back(tex);
        auto type = get_input2<std::string>("type");
        std::string suffix;
        if (type == "R") {
            suffix = ".x";
            type = "";
        }
        else if (type == "G") {
            suffix = ".y";
            type = "";
        }
        else if (type == "B") {
            suffix = ".z";
            type = "";
        }
        else if (type == "A") {
            suffix = ".w";
            type = "";
        }
        auto uvtiling = em->determineExpr(get_input("uvtiling").get());
        std::string coord = "att_uv";
        if (has_input("coord")) {
            coord = em->determineExpr(get_input("coord").get());
        }
        auto postprocess = get_input2<std::string>("post_process");
        if(postprocess == "raw"){
            em->emitCode(zeno::format("{}(texture2D(zenotex[{}], vec2({}) * {})){}", type, texId, coord, uvtiling, suffix));
        }else if (postprocess == "srgb"){
            em->emitCode(zeno::format("pow({}(texture2D(zenotex[{}], vec2({}) * {})),2.2f){}", type, texId, coord, uvtiling, suffix));
        }else if (postprocess == "normal_map"){
            em->emitCode(zeno::format("({}(texture2D(zenotex[{}], vec2({}) * {})) * 2.0f - 1.0f){}", type, texId, coord, uvtiling, suffix));
        }
    }
};

ZENDEFNODE(SmartTexture2D, {
    {
        {"readpath", "path"},
        {"heatmap"},
        {(std::string) "enum " + SmartTexture2D::texWrapping, "wrapS", "REPEAT"},
        {(std::string) "enum " + SmartTexture2D::texWrapping, "wrapT", "REPEAT"},
        {(std::string) "enum " + SmartTexture2D::texFiltering, "minFilter", "LINEAR"},
        {(std::string) "enum " + SmartTexture2D::texFiltering, "magFilter", "LINEAR"},
        {"coord"},
        {"vec2f", "uvtiling", "1,1"},
        {"vec4f", "value", "0,0,0,0"},
        {"enum float vec2 vec3 vec4 R G B A", "type", "vec3"},
        {"enum raw srgb normal_map", "post_process", "raw"}
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
