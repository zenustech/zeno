#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/utils/string.h>

namespace zeno {


static const char /* see https://docs.gl/sl4/trunc */
    unops[] = "copy neg abs sqrt inversesqrt exp log sin cos tan asin acos atan degrees"
              " radians sinh cosh tanh asinh acosh atanh round roundEven floor"
              " ceil trunc sign step length normalize",
    binops[] = "add sub mul div mod pow atan2 min max dot cross distance",
    ternops[] = "mix clamp smoothstep add3";


static auto &toHlsl() {
    static std::map<std::string, std::string> tab;
    return tab;
}


struct TreeTernaryMath : TreeNode {
    virtual int determineType(EmissionPass *em) override {
        auto op = get_input2<std::string>("op");
        auto in1 = get_input("in1");
        auto in2 = get_input("in2");
        auto in3 = get_input("in3");
        auto t1 = em->determineType(in1.get());
        auto t2 = em->determineType(in2.get());
        auto t3 = em->determineType(in3.get());

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

    virtual void emitCode(EmissionPass *em) override {
        auto op = get_input2<std::string>("op");
        auto in1 = em->determineExpr(get_input("in1").get(), this);
        auto in2 = em->determineExpr(get_input("in2").get(), this);
        auto in3 = em->determineExpr(get_input("in3").get(), this);

        if (op == "add3") {
            return em->emitCode(in1 + " + " + in2 + " + " + in3);
        } else {
            return em->emitCode(op + "(" + in1 + ", " + in2 + ", " + in3 + ")");
        }
    }
};

ZENDEFNODE(TreeTernaryMath, {
    {
        {"float", "in1", "0"},
        {"float", "in2", "0"},
        {"float", "in3", "0"},
        {(std::string)"enum " + ternops, "op", "mix"},
    },
    {
        {"float", "out"},
    },
    {},
    {"tree"},
});


struct TreeBinaryMath : TreeNode {
    virtual int determineType(EmissionPass *em) override {
        auto op = get_input2<std::string>("op");
        auto in1 = get_input("in1");
        auto in2 = get_input("in2");
        auto t1 = em->determineType(in1.get());
        auto t2 = em->determineType(in2.get());

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

    virtual void emitCode(EmissionPass *em) override {
        auto op = get_input2<std::string>("op");
        auto in1 = em->determineExpr(get_input("in1").get());
        auto in2 = em->determineExpr(get_input("in2").get());

        if (op == "add") {
            return em->emitCode(in1 + " + " + in2);
        } else if (op == "sub") {
            return em->emitCode(in1 + " - " + in2);
        } else if (op == "mul") {
            return em->emitCode(in1 + " * " + in2);
        } else if (op == "div") {
            return em->emitCode(in1 + " / " + in2);
        } else {
            return em->emitCode(op + "(" + in1 + ", " + in2 + ")");
        }
    }
};

ZENDEFNODE(TreeBinaryMath, {
    {
        {"float", "in1", "0"},
        {"float", "in2", "0"},
        {(std::string)"enum " + binops, "op", "add"},
    },
    {
        {"float", "out"},
    },
    {},
    {"tree"},
});


struct TreeUnaryMath : TreeNode {
    virtual int determineType(EmissionPass *em) override {
        auto op = get_input2<std::string>("op");
        auto in1 = get_input("in1");
        auto t1 = em->determineType(in1.get());

        return t1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto op = get_input2<std::string>("op");
        auto in1 = em->determineExpr(get_input("in1").get());

        if (op == "copy") {
            return em->emitCode(in1);
        } else if (op == "neg") {
            return em->emitCode("-" + in1);
        } else {
            return em->emitCode(op + "(" + in1 + ")");
        }
    }
};

ZENDEFNODE(TreeUnaryMath, {
    {
        {"float", "in1", "0"},
        {(std::string)"enum " + unops, "op", "sqrt"},
    },
    {
        {"float", "out"},
    },
    {},
    {"tree"},
});


}
