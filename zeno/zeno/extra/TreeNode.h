#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <vector>
#include <string>
#include <map>

namespace zeno {

struct EmissionPass;

struct TreeNode : INode {
    ZENO_API virtual void apply() override;
    ZENO_API virtual int determineType(EmissionPass *em) = 0;
    ZENO_API virtual void emitCode(EmissionPass *em) = 0;

    ZENO_API TreeNode();
    ZENO_API ~TreeNode();
};

struct EmissionPass {
    enum Backend {
        GLSL = 0,
        HLSL,  /* DID U KNOW THAT MICROSOFT BUYS GITHUB */
    };

    Backend backend = GLSL;

    struct VarInfo {
        int type;
        TreeNode *node;
    };

    struct CommonFunc {
        int rettype{};
        std::string name;
        std::vector<int> argTypes;
        std::string code;
    };

    std::map<TreeNode *, int> varmap;  /* varmap[node] = 40, then the variable of node is "tmp40" */
    std::vector<VarInfo> variables;  /* variables[40].type = 3, then the variable type will be "vec3 tmp40;" */
    std::vector<std::string> lines;  /* contains a list of operations, e.g. {"tmp40 = tmp41 + 1;", "tmp42 = tmp40 * 2;"} */
    std::vector<CommonFunc> commons;  /* definition of common functions, including custom functions and pre-defined functions */
    std::string commonCode;           /* other common codes written directly, e.g. "void myutilfunc() {...}" */
    std::string extensionsCode;       /* OpenGL extensions, e.g. "#extension GL_EXT_gpu_shader4 : enable" */

    ZENO_API std::string typeNameOf(int type) const;
    ZENO_API std::string finalizeCode(std::vector<std::pair<int, std::string>> const &keys,
                                      std::vector<std::shared_ptr<IObject>> const &vals);
    ZENO_API std::string finalizeCode();

    ZENO_API std::string addCommonFunc(CommonFunc func);
    ZENO_API std::string getCommonCode() const;

    ZENO_API std::string determineExpr(IObject *object) const;
    ZENO_API std::string determineExpr(IObject *object, TreeNode *node) const;
    ZENO_API std::string collectDefs() const;
    ZENO_API std::string collectCode() const;

    ZENO_API int currentType(TreeNode *node) const;
    ZENO_API int determineType(IObject *object);
    ZENO_API void emitCode(std::string const &line);
};

}
