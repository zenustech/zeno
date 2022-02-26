#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>

namespace zeno {

struct TreeBinaryOp : TreeNode {
    virtual void apply() override {
        settle("res", {"lhs", "rhs"}, {"op"});
    }
};

ZENDEFNODE(TreeBinaryOp, {
    {
        {"float", "lhs"},
        {"float", "rhs"},
    },
    {
        {"tree", "res"},
    },
    {
        {"enum add sub div mul", "op", "add"},
    },
    {"tree"},
});

}
