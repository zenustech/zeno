#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno
{
    // reference: https://github.com/jamieowen/glsl-blend
    static constexpr char blend_mode_str[] =
        "add" " "
        "average" " "
        "colorBurn" " "
        "colorDodge" " "
        "darken" " "
        "difference" " "
        "exclusion" " "
        "glow" " "
        "hardLight" " "
        "hardMix" " "
        "lighten" " "
        "linearBurn" " "
        "linearDodge" " "
        "linearLight" " "
        "multiply" " "
        "negation" " "
        "normal" " "
        "overlay" " "
        "phoenix" " "
        "pinLight" " "
        "reflect" " "
        "screen" " "
        "softLight" " "
        "subtract" " "
        "vividLight" " "
    ;

    static const std::unordered_map<std::string, std::string> blend_func_code = {
        {
            "add",
            "return (min(base + blend, vec3(1.0)) * opacity + base * (1.0 - opacity));"
        },
        {  
            "average",
            "return ((base + blend) / 2.0 * opacity + base * (1.0 - opacity));"
        },
        {   
            "colorBurn", 
            "float r = (blend.r == 0.0) ? blend.r : max((1.0 - ((1.0 - base.r) / blend.r)), 0.0);\n"
            "float g = (blend.g == 0.0) ? blend.g : max((1.0 - ((1.0 - base.g) / blend.g)), 0.0);\n"
            "float b = (blend.b == 0.0) ? blend.b : max((1.0 - ((1.0 - base.b) / blend.b)), 0.0);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {   
            "colorDodge", 
            "float r = (blend.r == 1.0) ? blend.r : min(base.r / (1.0 - blend.r), 1.0);\n"
            "float g = (blend.g == 1.0) ? blend.g : min(base.g / (1.0 - blend.g), 1.0);\n"
            "float b = (blend.b == 1.0) ? blend.b : min(base.b / (1.0 - blend.b), 1.0);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {   
            "darken", 
            "float r = min(blend.r, base.r);\n"
            "float g = min(blend.g, base.g);\n"
            "float b = min(blend.b, base.b);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {   
            "difference", 
            "return (abs(base - blend) * opacity + base * (1.0 - opacity));"
        },
        {   
            "exclusion", 
            "return ((base + blend - 2.0 * base * blend) * opacity + base * (1.0 - opacity));"
        },
        {
            "glow",
            "float r = (blend.r == 1.0) ? blend.r : min(base.r * base.r / (1.0 - blend.r), 1.0);\n"
            "float g = (blend.g == 1.0) ? blend.g : min(base.g * base.g / (1.0 - blend.g), 1.0);\n"
            "float b = (blend.b == 1.0) ? blend.b : min(base.b * base.b / (1.0 - blend.b), 1.0);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "hardLight",
            "float r = base.r < 0.5 ? (2.0 * base.r * blend.r) : (1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r));\n"
            "float g = base.g < 0.5 ? (2.0 * base.g * blend.g) : (1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g));\n"
            "float b = base.b < 0.5 ? (2.0 * base.r * blend.b) : (1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b));\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "hardMix",
            "float r = 0.0;\n"
            "if (blend.r < 0.5) {\n"
            "   float tempBlendR = 2.0 * blend.r;\n"
            "   r = (tempBlendR == 0.0) ? tempBlendR : max((1.0 - ((1.0 - base.r) / tempBlendR)), 0.0);\n"
            "} else {\n"
            "   float tempBlendR = 2.0 * (blend.r - 0.5);\n"
            "   r = (tempBlendR == 1.0) ? tempBlendR : min(base.r / (1.0 - tempBlendR), 1.0);\n"
            "}\n"
            "r = (r < 0.5) ? 0.0 : 1.0;\n"
            "float g = 0.0;\n"
            "if (blend.g < 0.5) {\n"
            "   float tempBlendG = 2.0 * blend.g;\n"
            "   g = (tempBlendG == 0.0) ? tempBlendG : max((1.0 - ((1.0 - base.g) / tempBlendG)), 0.0);\n"
            "} else {\n"
            "   float tempBlendG = 2.0 * (blend.g - 0.5);\n"
            "   g = (tempBlendG == 1.0) ? tempBlendG : min(base.g / (1.0 - tempBlendG), 1.0);\n"
            "}\n"
            "g = (g < 0.5) ? 0.0 : 1.0;\n"
            "float b = 0.0;\n"
            "if (blend.b < 0.5) {\n"
            "   float tempBlendB = 2.0 * blend.b;\n"
            "   b = (tempBlendB == 0.0) ? tempBlendB : max((1.0 - ((1.0 - base.b) / tempBlendB)), 0.0);\n"
            "} else {\n"
            "   float tempBlendB = 2.0 * (blend.b - 0.5);\n"
            "   b = (tempBlendB == 1.0) ? tempBlendB : min(base.b / (1.0 - tempBlendB), 1.0);\n"
            "}\n"
            "b = (b < 0.5) ? 0.0 : 1.0;\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "lighten",
            "float r = max(blend.r, base.r);\n"
            "float g = max(blend.g, base.g);\n"
            "float b = max(blend.b, base.b);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "linearBurn",
            "return (max(base + blend - vec3(1.0), vec3(1.0)) * opacity + base * (1.0 - opacity));"
        },
        {
            "linearDodge",
            "return (min(base + blend, vec3(1.0)) * opacity + base * (1.0 - opacity));"
        },
        {
            "linearLight",
            "float r = blend.r < 0.5 ? max(base.r + 2.0 * blend.r - 1.0, 0.0) : min(base.r + 2.0 * (blend.r - 0.5), 1.0);\n"
            "float g = blend.g < 0.5 ? max(base.g + 2.0 * blend.g - 1.0, 0.0) : min(base.g + 2.0 * (blend.g - 0.5), 1.0);\n"
            "float b = blend.b < 0.5 ? max(base.b + 2.0 * blend.b - 1.0, 0.0) : min(base.b + 2.0 * (blend.b - 0.5), 1.0);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "multiply",
            "return (base * blend * opacity + base * (1.0 - opacity));"
        },
        {
            "negation",
            "return ((vec3(1.0) - abs(vec3(1.0) - base - blend)) * opacity + base * (1.0 - opacity));"
        },
        {
            "normal",
            "return (blend * opacity + base * (1.0 - opacity));"
        },
        {
            "overlay",
            "float r = base.r < 0.5 ? (2.0 * base.r * blend.r) : (1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r));\n"
            "float g = base.g < 0.5 ? (2.0 * base.g * blend.g) : (1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g));\n"
            "float b = base.b < 0.5 ? (2.0 * base.r * blend.b) : (1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b));\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "phoenix",
            "return ((min(base, blend) - max(base, blend) + vec3(1.0)) * opacity + base * (1.0 - opacity));"
        },
        {
            "pinLight",
            "float r = base.r < 0.5 ? min(blend.r, base.r) : max(blend.r, base.r);\n"
            "float g = base.g < 0.5 ? min(blend.g, base.g) : max(blend.g, base.g);\n"
            "float b = base.b < 0.5 ? min(blend.b, base.b) : max(blend.b, base.b);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "reflect",
            "float r = (blend.r == 1.0) ? blend.r : min(base.r * base.r / (1.0 - blend.r), 1.0);\n"
            "float g = (blend.g == 1.0) ? blend.g : min(base.g * base.g / (1.0 - blend.g), 1.0);\n"
            "float b = (blend.b == 1.0) ? blend.b : min(base.b * base.b / (1.0 - blend.b), 1.0);\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "screen",
            "float r = 1.0 - ((1.0 - base.r) * (1.0 - blend.r));\n"
            "float g = 1.0 - ((1.0 - base.g) * (1.0 - blend.g));\n"
            "float b = 1.0 - ((1.0 - base.b) * (1.0 - blend.b));\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "softLight",
            "float r = (blend.r < 0.5) ? (2.0 * base.r * blend.r + base.r * base.r * (1.0 - 2.0 * blend.r)) : (sqrt(base.r) * (2.0 * blend.r - 1.0) + 2.0 * base.r * (1.0 - blend.r));\n"
            "float g = (blend.g < 0.5) ? (2.0 * base.g * blend.g + base.g * base.g * (1.0 - 2.0 * blend.g)) : (sqrt(base.g) * (2.0 * blend.g - 1.0) + 2.0 * base.g * (1.0 - blend.g));\n"
            "float b = (blend.b < 0.5) ? (2.0 * base.b * blend.b + base.b * base.b * (1.0 - 2.0 * blend.b)) : (sqrt(base.b) * (2.0 * blend.b - 1.0) + 2.0 * base.b * (1.0 - blend.b));\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
        {
            "subtract",
            "return (max(base + blend - vec3(1.0), vec3(0.0)) * opacity + base * (1.0 - opacity));"
        },
        {
            "vividLight",
            "float r = 0.0;\n"
            "if (blend.r < 0.5) {\n"
            "   float tempBlendR = 2.0 * blend.r;\n"
            "   r = (tempBlendR == 0.0) ? tempBlendR : max((1.0 - ((1.0 - base.r) / tempBlendR)), 0.0);\n"
            "} else {\n"
            "   float tempBlendR = 2.0 * (blend.r - 0.5);\n"
            "   r = (tempBlendR == 1.0) ? tempBlendR : min(base.r / (1.0 - tempBlendR), 1.0);\n"
            "}\n"
            "float g = 0.0;\n"
            "if (blend.g < 0.5) {\n"
            "   float tempBlendG = 2.0 * blend.g;\n"
            "   g = (tempBlendG == 0.0) ? tempBlendG : max((1.0 - ((1.0 - base.g) / tempBlendG)), 0.0);\n"
            "} else {\n"
            "   float tempBlendG = 2.0 * (blend.g - 0.5);\n"
            "   g = (tempBlendG == 1.0) ? tempBlendG : min(base.g / (1.0 - tempBlendG), 1.0);\n"
            "}\n"
            "float b = 0.0;\n"
            "if (blend.b < 0.5) {\n"
            "   float tempBlendB = 2.0 * blend.b;\n"
            "   b = (tempBlendB == 0.0) ? tempBlendB : max((1.0 - ((1.0 - base.b) / tempBlendB)), 0.0);\n"
            "} else {\n"
            "   float tempBlendB = 2.0 * (blend.b - 0.5);\n"
            "   b = (tempBlendB == 1.0) ? tempBlendB : min(base.b / (1.0 - tempBlendB), 1.0);\n"
            "}\n"
            "return (vec3(r, g, b) * opacity + base * (1.0 - opacity));"
        },
    };

    struct ShaderBlendMode
        : ShaderNodeClone<ShaderBlendMode>
    {
        virtual int determineType(EmissionPass *em) override
        {
            auto base = get_input("base");
            auto tBase = em->determineType(base.get());
            if (tBase != 3) throw zeno::Exception("base's dimension mismatch: " + std::to_string(tBase));

            auto blend = get_input("blend");
            auto tBlend = em->determineType(blend.get());
            if (tBlend != 3) throw zeno::Exception("blend's dimension mismatch: " + std::to_string(tBlend));

            auto opacity = get_input("opacity");
            auto tOpacity = em->determineType(opacity.get());
            if (tOpacity != 1) throw zeno::Exception("opacity's dimension mismatch: " + std::to_string(tOpacity));

            return 3;
        }

        virtual void emitCode(EmissionPass *em) override
        {
            auto mode = get_input2<std::string>("mode");
            auto base = em->determineExpr(get_input("base").get(), this);
            auto blend = em->determineExpr(get_input("blend").get(), this);
            auto opacity = em->determineExpr(get_input("opacity").get());

            EmissionPass::CommonFunc comm
            {
                rettype : 3,
                argTypes : {3, 3, 1},
                code : "(vec3 base, vec3 blend, float opacity) {\n" + blend_func_code.at(mode) + "\n}",
            };
            auto blendFuncName = em->addCommonFunc(std::move(comm));

            return em->emitCode(blendFuncName + "(" + base + ", " + blend + ", " + opacity + ")");
        }
    };

    ZENDEFNODE(
        ShaderBlendMode,
        {
            {
                {"float", "base", "0"},
                {"float", "blend", "0"},
                {"float", "opacity", "0"},
                {(std::string) "enum " + blend_mode_str, "mode", "add"},
            },
            {
                {"shader", "out"},
            },
            {},
            {"shader"},
        });
} // namespace zeno