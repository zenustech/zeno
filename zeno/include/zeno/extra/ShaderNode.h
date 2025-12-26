#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/TextureObject.h>
#include <tinygltf/json.hpp>
#include <vector>
#include <string>
#include <map>

namespace zeno {

struct EmissionPass;

struct ShaderNode : INode {
    ZENO_API virtual void apply() override;
    ZENO_API virtual int determineType(EmissionPass *em) = 0;
    ZENO_API virtual void emitCode(EmissionPass *em) = 0;
    ZENO_API virtual std::shared_ptr<ShaderNode> clone() const = 0;

    ZENO_API ShaderNode();
    ZENO_API ~ShaderNode() override;
};

inline const auto ShaderDataTypeNames = 
                        std::array { "bool", "int", "int3", "int4", "uint", "uint3", "uint4", "int64_t", "uint64_t", "float", "vec2", "vec3", "vec4" };

using ShaderDataTypeList = std::tuple<bool, int32_t, vec3i, vec4i, uint32_t, vec3I, vec4I, int64_t, uint64_t, float, vec2f, vec3f, vec4f>;

template<typename T, std::size_t I = 0>
static std::string dispatchTypeName() {
    if constexpr (I == std::tuple_size_v<ShaderDataTypeList>) {
        return ""; // end of loop
    } else {
        using Type = std::tuple_element_t<I, ShaderDataTypeList>;
        if constexpr (std::is_same_v<Type, T>) {
            return ShaderDataTypeNames[I];
        }
        return dispatchTypeName<T, I+1>(); // next iteration
    }
}

template<typename T>
static std::string typeNameStatic() {
    return dispatchTypeName<T>();
}

static const inline std::string ShaderDataTypeNamesString = []() {
    const auto& names = ShaderDataTypeNames;
    std::string result;
    result += names[0];
    for (int i=1; i<names.size(); ++i) {
        result += std::string(" ") + names[i];
    }
    return result;
} ();

static const inline std::map<std::string, int> TypeHint {

    {"bool", 0},
    {"int", 11},
    {"int3", 13},
    {"int4", 14},

    {"uint", 21},
    {"uint3", 23},   
    {"uint4", 24},

    {"int64_t", 31},
    {"uint64_t", 41},

    {"float", 1},
    {"vec2", 2},
    {"vec3", 3},
    {"vec4", 4}
};

static const inline std::map<int, std::string> TypeHintReverse = []() {

    std::map<int, std::string> result {};
    for (auto& [k, v] : TypeHint) {
        result[v] = k;
    }
    return result;
} ();

template <class Derived>
struct ShaderNodeClone : ShaderNode {
    virtual std::shared_ptr<ShaderNode> clone() const override {
        return std::make_shared<Derived>(static_cast<Derived const &>(*this));
    }
};

struct EmissionPass {
    enum Backend {
        GLSL = 0,
        HLSL,  /* DID U KNOW THAT MICROSOFT BUYS GITHUB */
    };

    Backend backend = GLSL;

    struct ConstInfo {
        int type;
        NumericValue value;
    };

    struct VarInfo {
        int type;
        ShaderNode *node;
    };

    struct CommonFunc {
        int rettype{};
        std::string name;
        std::vector<int> argTypes;
        std::string code;
    };

    std::map<NumericObject *, int> constmap;
    std::vector<ConstInfo> constants;
    std::map<ShaderNode *, int> varmap;  /* varmap[node] = 40, then the variable of node is "tmp40" */
    std::vector<VarInfo> variables;  /* variables[40].type = 3, then the variable type will be "vec3 tmp40;" */
    std::vector<std::string> lines;  /* contains a list of operations, e.g. {"tmp40 = tmp41 + 1;", "tmp42 = tmp40 * 2;"} */
    std::vector<CommonFunc> commons;  /* definition of common functions, including custom functions and pre-defined functions */
    std::string commonCode;           /* other common codes written directly in GLSL, e.g. "void myutilfunc() {...}" */
    std::string extensionsCode;       /* OpenGL extensions, e.g. "#extension GL_EXT_gpu_shader4 : enable" */
    std::vector<std::shared_ptr<Texture2DObject>> tex2Ds;

    ZENO_API std::string typeNameOf(int type) const;
    ZENO_API std::string funcName(std::string const &fun) const;

    ZENO_API std::string finalizeCode(std::vector<std::pair<int, std::string>> const &keys,
                                      std::vector<std::shared_ptr<IObject>> const &vals);
    ZENO_API std::string finalizeCode();

    ZENO_API std::string addCommonFunc(CommonFunc func);
    ZENO_API std::string getCommonCode() const;

    ZENO_API void duplicateIfHlsl(int type, std::string &expr) const;
    ZENO_API static void translateToHlsl(std::string &code);
    ZENO_API void translateCommonCode();

    ZENO_API std::string determineExpr(IObject *object) const;
    ZENO_API std::string determineExpr(IObject *object, ShaderNode *node) const;
    ZENO_API std::string collectDefs() const;
    ZENO_API std::string collectCode() const;

    ZENO_API int currentType(ShaderNode *node) const;
    ZENO_API int determineType(IObject *object);
    ZENO_API void emitCode(std::string const &line);
};

}
