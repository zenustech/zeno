#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include <algorithm>

namespace zeno
{
struct ShaderTexture2D : ShaderNodeClone<ShaderTexture2D>
{
    virtual int determineType(EmissionPass *em) override {
        auto texId = get_input2<int>("texId");
        auto coord = em->determineType(get_input("coord").get());
        if (coord < 2)
            throw zeno::Exception("ShaderTexture2D expect coord to be at least vec2");

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
        auto coord = em->determineExpr(get_input("coord").get());
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto ty = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
        const char *bat[] = {"x", "xy", "xyz", "xyzw"};
        int isCuda = 0;
        if(has_input("isCuda"))
            isCuda = get_input2<int>("isCuda");
        std::string xy = ".xy).";
        if(isCuda==1)
            xy = ".xy()).";
        std::string vec = bat[ty];
        if(isCuda==1&&ty>0)
            vec = vec + "()";
        em->emitCode("texture2D(zenotex" + std::to_string(texId) + ", " + coord + xy + vec);
    }
};

ZENDEFNODE(ShaderTexture2D, {
    {
        {"int", "isCuda", "0"},
        {"int", "texId", "0"},
        {"vec2f", "coord", "0,0"},
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
