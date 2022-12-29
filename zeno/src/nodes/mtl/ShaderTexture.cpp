#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include <algorithm>
#include "zeno/utils/format.h"

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
        em->emitCode(zeno::format("{}(texture2D(zenotex{}, vec2({}) * {}))", type, texId, coord, uvtiling));
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

} // namespace zeno
