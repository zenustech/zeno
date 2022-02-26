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
    struct VarInfo {
        int type;
        TreeNode *node;
    };

    struct FinalCode {
        std::string code;
    };

    std::map<TreeNode *, int> varmap;  // varmap[node] = 40, then the variable of node is "tmp40"
    std::vector<VarInfo> variables;  // variables[40].type = 3, then the variable type will be "vec3 tmp40;"
    std::vector<std::string> lines;  // contains a list of operations, e.g. {"tmp40 = tmp41 + 1;", "tmp42 = tmp40 * 2;"}

    ZENO_API FinalCode finalizeOutput(IObject *object);

    ZENO_API std::string determineExpr(IObject *object) const;
    ZENO_API std::string collectDefs() const;
    ZENO_API std::string collectCode() const;

    ZENO_API int determineType(IObject *object);
    ZENO_API void emitCode(std::string const &line);
};

}
