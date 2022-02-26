#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/utils/string.h>

namespace zeno {

static const char /* see https://docs.gl/sl4/trunc */
    unops[] = "abs sqrt inversesqrt exp log sin cos tan asin acos atan degrees radians sinh cosh tanh asinh acosh atanh round roundEven floor ceil trunc sign step length normalize",
    binops[] = "add sub mul div mod pow atan2 min max dot cross distance",
    ternops[] = "mix clamp smoothstep";

struct TreeTernaryMath : TreeNode {
    virtual int determineType() override {
        auto op = get_param<std::string>("op");
        auto in1 = get_input("in1");
        auto in2 = get_input("in2");
        auto t1 = determineTypeOf(in1);
        auto t2 = determineTypeOf(in2);

        if (t1 == 1 && t2 == t3) {
            return t2;
        } else if (t2 == 1 && t3 == t1) {
            return t3;
        } else if (t3 == 1 && t1 == t2) {
            return t1;
        } else if (t1 == 1 && t2 == 1) {
            return t3;
        } else if (t2 == 1 && t3 == 1) {
            return t2;
        } else if (t3 == 1 && t1 == 1) {
            return t2;
        } else if (t1 == t2 && t2 == t3) {
            return t1;
        } else {
            throw zeno::Exception("vector dimension mismatch: " + std::to_string(t1) + ", " + std::to_string(t2) + ", " + std::to_string(t3));
        }
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
    virtual int determineType() override {
        auto op = get_param<std::string>("op");
        auto in1 = get_input("in1");
        auto in2 = get_input("in2");
        auto t1 = determineTypeOf(in1);
        auto t2 = determineTypeOf(in2);

        if (op == "dot") {
            if (t1 != t2)
                throw zeno::Exception("both-side of dot must have same dimension");
            else if (t1 == 1)
                throw zeno::Exception("dot only work for vectors");
            else
                return 1;

        } else if (op == "cross") {
            if (t1 != t2)
                throw zeno::Exception("both-side of cross must have same dimension");
            else if (t1 == 2)
                return 1;
            else if (t1 == 3)
                return 3;
            else
                throw zeno::Exception("dot only work for 2d and 3d vectors");

        } else if (op == "distance") {
            if (t1 != t2)
                throw zeno::Exception("both-side of distance must have same dimension");
            else if (t1 == 1)
                throw zeno::Exception("distance only work for vectors");
            else
                return t1;

        } else if (t1 == 1) {
            return t2;
        } else if (t2 == 1) {
            return t1;
        } else if (t1 == t2) {
            return t1;
        } else {
            throw zeno::Exception("vector dimension mismatch: " + std::to_string(t1) + " != " + std::to_string(t2));
        }
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
    virtual int determineType() override {
        auto op = get_param<std::string>("op");
        auto in1 = get_input("in1");
        auto t1 = determineTypeOf(in1);

        return t1;
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
