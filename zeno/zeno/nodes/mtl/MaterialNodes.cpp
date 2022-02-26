#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <set>

namespace zeno {

static const char /* see https://docs.gl/sl4/trunc */
    unops[] = "abs sqrt inversesqrt exp log sin cos tan asin acos atan degrees radians sinh cosh tanh asinh acosh atanh round roundEven floor ceil trunc sign step length normalize",
    binops[] = "add sub mul div mod pow atan2 min max dot cross distance",
    ternops[] = "mix clamp smoothstep refract";

struct TreeTernaryMath : TreeNode {
    virtual void apply() override {
        auto op = get_param<std::string>("op");
        settle("out", {"in1", "in2", "in3"}, {"op"});
    }
};

ZENDEFNODE(TreeTernaryMath, {
    {
        {"float", "in1"},
        {"float", "in2"},
        {(std::string)"enum " + ternops, "op", "add"},
    },
    {
        {"float", "out"},
    },
    {},
    {"tree"},
});

struct TreeBinaryMath : TreeNode {
    virtual void apply() override {
        auto op = get_param<std::string>("op");
        settle("out", {"in1", "in2"}, {"op"});
    }
};

ZENDEFNODE(TreeBinaryMath, {
    {
        {"float", "in1"},
        {"float", "in2"},
        {(std::string)"enum " + binops, "op", "add"},
    },
    {
        {"float", "out"},
    },
    {},
    {"tree"},
});

struct TreeUnaryMath : TreeNode {
    virtual void apply() override {
        auto op = get_param<std::string>("op");
        settle("out", {"in1"}, {"op"});
    }
};

ZENDEFNODE(TreeUnaryMath, {
    {
        {"float", "in1"},
        {"float", "in2"},
        {(std::string)"enum " + unops, "op", "add"},
    },
    {
        {"float", "out"},
    },
    {},
    {"tree"},
});

}
