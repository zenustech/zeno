#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno
{
    // reference: https://github.com/jamieowen/glsl-blend
    static const std::string BLEND_ENUM_STR =
        "enum"
        " " "add"
        " " "average"
        " " "colorBurn"
        " " "colorDodge"
        " " "darken"
        " " "difference"
        " " "exclusion"
        " " "glow"
        " " "hardLight"
        " " "hardMix"
        " " "lighten"
        " " "linearBurn"
        " " "linearDodge"
        " " "linearLight"
        " " "multiply"
        " " "negation"
        " " "normal"
        " " "overlay"
        " " "phoenix"
        " " "pinLight"
        " " "reflect"
        " " "screen"
        " " "softLight"
        " " "subtract"
        " " "vividLight"
    ; // BLEND_ENUM_STR

    static const std::unordered_map<std::string, std::string> BLEND_CODE_MAP =
    {
        {
            "add",
            "   return min(base + blend, 1.0);\n"
        },
        {  
            "average",
            "   return (base + blend) / 2.0;\n"
        },
        {   
            "colorBurn", 
            "   return (blend == 0.0) ? blend : max((1.0 - ((1.0 - base) / blend)), 0.0);\n"
        },
        {   
            "colorDodge", 
            "   return = (blend == 1.0) ? blend : min(base / (1.0 - blend), 1.0);\n"
        },
        {   
            "darken", 
            "   return min(blend, base);\n"
        },
        {   
            "difference", 
            "   return abs(base - blend);\n"
        },
        {   
            "exclusion", 
            "   return (base + blend - 2.0 * base * blend);\n"
        },
        {
            "glow",
            "   return (blend == 1.0) ? blend : min(base * base / (1.0 - blend), 1.0);\n"
        },
        {
            "hardLight",
            "   return base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend));\n"
        },
        {
            "hardMix",
            "   if (blend < 0.5) {\n"
            "       blend = 2.0 * blend;\n"
            "       blend = (blend == 0.0) ? blend : max((1.0 - ((1.0 - base) / blend)), 0.0);\n"
            "   } else {\n"
            "       blend = 2.0 * (blend - 0.5);\n"
            "       blend = (blend == 1.0) ? blend : min(base / (1.0 - blend), 1.0);\n"
            "   }\n"
            "   return (blend < 0.5) ? 0.0 : 1.0;\n"
        },
        {
            "lighten",
            "   return max(blend, base);\n"
        },
        {
            "linearBurn",
            "   return max(base + blend - 1.0, 1.0);\n"
        },
        {
            "linearDodge",
            "   return min(base + blend, 1.0);\n"
        },
        {
            "linearLight",
            "   return (blend < 0.5) ? max(base + 2.0 * blend - 1.0, 0.0) : min(base + 2.0 * (blend - 0.5), 1.0);\n"
        },
        {
            "multiply",
            "   return base * blend;\n"
        },
        {
            "negation",
            "   return (1.0 - abs(1.0 - base - blend));\n"
        },
        {
            "normal",
            "   return blend;\n"
        },
        {
            "overlay",
            "   return (base < 0.5) ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend));\n"
        },
        {
            "phoenix",
            "   return (min(base, blend) - max(base, blend) + 1.0)\n;"
        },
        {
            "pinLight",
            "   return (base < 0.5) ? min(blend, base) : max(blend, base);\n"
        },
        {
            "reflect",
            "   return (blend == 1.0) ? blend : min(base * base / (1.0 - blend), 1.0);\n"
        },
        {
            "screen",
            "   return 1.0 - ((1.0 - base) * (1.0 - blend));\n"
        },
        {
            "softLight",
            "   return (blend < 0.5) ? (2.0 * base * blend + base * base * (1.0 - 2.0 * blend)) : (sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend));\n"
        },
        {
            "subtract",
            "   return max(base + blend - 1.0, 0.0);\n"
        },
        {
            "vividLight",
            "   if (blend < 0.5) {\n"
            "       blend = 2.0 * blend;\n"
            "       return (blend == 0.0) ? blend : max((1.0 - ((1.0 - base) / blend)), 0.0);\n"
            "   } else {\n"
            "       blend = 2.0 * (blend - 0.5);\n"
            "       return (blend == 1.0) ? blend : min(base / (1.0 - blend), 1.0);\n"
            "   }\n"
        },
    }; // BLEND_CODE_MAP

    struct ShaderBlendMode
        : ShaderNodeClone<ShaderBlendMode>
    {
        virtual int determineType(EmissionPass *em) override
        {
            const auto base = get_input("base");
            const auto baseDim = em->determineType(base.get());
            const auto blend = get_input("blend");
            const auto blendDim = em->determineType(blend.get());
            if (baseDim != blendDim)
                throw zeno::Exception("base and blend's dimension mismatch: " + std::to_string(baseDim) + ", " + std::to_string(blendDim));

            const auto opacity = get_input("opacity");
            const auto opacityDim = em->determineType(opacity.get());
            if (opacityDim != 1)
                throw zeno::Exception("opacity's dimension mismatch: " + std::to_string(opacityDim));

            return baseDim;
        } // determineType

        virtual void emitCode(EmissionPass *em) override
        {
            const auto mode = get_input2<std::string>("mode");
            const auto mapIter = BLEND_CODE_MAP.find(mode);
            if (mapIter == std::end(BLEND_CODE_MAP))
            {
                throw zeno::Exception("mode mismatch: " + mode);
            }
            const auto& code = mapIter->second;

            EmissionPass::CommonFunc blendFunc;
            blendFunc.rettype = 1;
            blendFunc.argTypes = {1, 1};
            blendFunc.code = "(" + em->typeNameOf(1) + " base, " + em->typeNameOf(1) + " blend) {\n" + code + "}\n";
            const auto blendFuncName = em->addCommonFunc(std::move(blendFunc));

            const auto base = get_input("base");
            const auto baseDim = em->determineType(base.get());

            EmissionPass::CommonFunc opacityFunc;
            switch (baseDim)
            {
            case 1:
                opacityFunc.rettype = 1;
                opacityFunc.argTypes = {1, 1, 1};
                opacityFunc.code = 
                "(" + em->typeNameOf(1) + " base, " + em->typeNameOf(1) + " blend, " + em->typeNameOf(1) + " opacity) {\n"
                "   blend = " + blendFuncName + "(base, blend);\n"
                "   return blend * opacity + base * (1.0 - opacity);\n"
                "}\n";
                break;
            case 2:
                opacityFunc.rettype = 2;
                opacityFunc.argTypes = {2, 2, 1};
                opacityFunc.code = 
                "(" + em->typeNameOf(2) + " base, " + em->typeNameOf(2) + " blend, " + em->typeNameOf(1) + " opacity) {\n"
                "   blend.r = " + blendFuncName + "(base.r, blend.r);\n"
                "   blend.g = " + blendFuncName + "(base.g, blend.g);\n"
                "   return blend * opacity + base * (1.0 - opacity);\n"
                "}\n";
                break;
            case 3:
                opacityFunc.rettype = 3;
                opacityFunc.argTypes = {3, 3, 1};
                opacityFunc.code = 
                "(" + em->typeNameOf(3) + " base, " + em->typeNameOf(3) + " blend, " + em->typeNameOf(1) + " opacity) {\n"
                "   blend.r = " + blendFuncName + "(base.r, blend.r);\n"
                "   blend.g = " + blendFuncName + "(base.g, blend.g);\n"
                "   blend.b = " + blendFuncName + "(base.b, blend.b);\n"
                "   return blend * opacity + base * (1.0 - opacity);\n"
                "}\n";
                break;
            case 4:
                opacityFunc.rettype = 4;
                opacityFunc.argTypes = {4, 4, 1};
                opacityFunc.code = 
                "(" + em->typeNameOf(4) + " base, " + em->typeNameOf(4) + " blend, " + em->typeNameOf(1) + " opacity) {\n"
                "   blend.r = " + blendFuncName + "(base.r, blend.r);\n"
                "   blend.g = " + blendFuncName + "(base.g, blend.g);\n"
                "   blend.b = " + blendFuncName + "(base.b, blend.b);\n"
                "   blend.a = " + blendFuncName + "(base.a, blend.a);\n"
                "   return blend * opacity + base * (1.0 - opacity);\n"
                "}\n";
                break;
            default:
                throw zeno::Exception("base's dimension mismatch: " + std::to_string(baseDim));
                break;
            }
            const auto opacityFuncName = em->addCommonFunc(std::move(opacityFunc));

            const auto baseExpr = em->determineExpr(base.get());
            const auto blend = get_input("blend");
            const auto blendExpr = em->determineExpr(blend.get());
            const auto opacity = get_input("opacity");
            const auto opacityExpr = em->determineExpr(opacity.get());
            return em->emitCode(opacityFuncName + "(" + baseExpr + ", " + blendExpr + ", " + opacityExpr + ")");
        } // emitCode
    }; // struct ShaderBlendMode

    ZENDEFNODE(
        ShaderBlendMode,
        {
            {
                {"float", "base", "0"},
                {"float", "blend", "0"},
                {"float", "opacity", "0"},
                {BLEND_ENUM_STR, "mode", "normal"},
            },
            {
                {"shader", "out"},
            },
            {},
            {"shader"},
        });
} // namespace zeno