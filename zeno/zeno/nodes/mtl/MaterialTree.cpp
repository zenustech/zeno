#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/utils/string.h>

namespace zeno {

static const char /* see https://docs.gl/sl4/trunc */
    unops[] = "copy neg abs sqrt inversesqrt exp log sin cos tan asin acos atan degrees"
              " radians sinh cosh tanh asinh acosh atanh round roundEven floor"
              " ceil trunc sign step length normalize",
    binops[] = "add sub mul div mod pow atan2 min max dot cross distance",
    ternops[] = "mix clamp smoothstep";


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
        auto in1 = em->determineExpr(get_input("in1").get());
        auto in2 = em->determineExpr(get_input("in2").get());
        auto in3 = em->determineExpr(get_input("in3").get());

        em->emitCode(op + "(" + in1 + ", " + in2 + ", " + in3 + ")");
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


struct TreeFinalize : INode {
    virtual void apply() override {
        auto code = EmissionPass{}.finalizeCode({
            "mat_basecolor",
            "mat_metallic",
            "mat_roughness",
            "mat_specular",
            "mat_normal",
            "mat_emission",
            "mat_emitrate",
        }, {
            get_input<IObject>("basecolor", std::make_shared<NumericObject>(vec3f(1.0f))),
            get_input<IObject>("metallic", std::make_shared<NumericObject>(float(0.0f))),
            get_input<IObject>("roughness", std::make_shared<NumericObject>(float(0.4f))),
            get_input<IObject>("specular", std::make_shared<NumericObject>(float(0.5f))),
            get_input<IObject>("normal", std::make_shared<NumericObject>(vec3f(0, 0, 1))),
            get_input<IObject>("emission", std::make_shared<NumericObject>(vec3f(0))),
            get_input<IObject>("emitrate", std::make_shared<NumericObject>(float(0.f))),
        });
        set_output2("code", code);
    }
};

ZENDEFNODE(TreeFinalize, {
    {
        {"vec3f", "basecolor", "1,1,1"},
        {"float", "metallic", "0.0"},
        {"float", "roughness", "0.4"},
        {"float", "specular", "0.5"},
        {"vec3f", "normal", "0,0,1"},
        {"vec3f", "emission", "0,0,0"},
        {"float", "emitrate", "0"},
    },
    {
        {"string", "code"},
    },
    {},
    {"tree"},
});


struct TreeInputAttr : TreeNode {
    virtual int determineType(EmissionPass *em) override {
        auto attr = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), attr) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto attr = get_input2<std::string>("attr");
        return em->emitCode("att_" + attr);
    }
};

ZENDEFNODE(TreeInputAttr, {
    {
        {"enum pos clr nrm", "attr", "pos"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"tree", "out"},
    },
    {},
    {"tree"},
});


struct TreeLinearFit : TreeNode {
    virtual int determineType(EmissionPass *em) override {
        auto in = em->determineType(get_input("in").get());
        auto inMin = em->determineType(get_input("inMin").get());
        auto inMax = em->determineType(get_input("inMax").get());
        auto outMin = em->determineType(get_input("outMin").get());
        auto outMax = em->determineType(get_input("outMax").get());

        if (inMin == 1 && inMax == 1 && outMin == 1 && outMax == 1) {
            return in;
        } else if (inMin == in && inMax == in && outMin == in && outMax == in) {
            return in;
        } else if (inMin == 1 && inMax == 1 && outMin == in && outMax == in) {
            return in;
        } else if (inMin == in && inMax == in && outMin == 1 && outMax == 1) {
            return in;
        } else {
            throw zeno::Exception("vector dimension mismatch in linear fit");
        }
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(get_input("in").get());
        auto inMin = em->determineExpr(get_input("inMin").get());
        auto inMax = em->determineExpr(get_input("inMax").get());
        auto outMin = em->determineExpr(get_input("outMin").get());
        auto outMax = em->determineExpr(get_input("outMax").get());

        auto exp = "(" + in + " - " + inMin + ") / (" + inMax + " - " + inMin + ")";
        if (get_param<bool>("clamped"))
            exp = "clamp(" + exp + ", 0.0, 1.0)";
        em->emitCode(exp + " * (" + outMax + " - " + outMin + ") + " + outMax);
    }
};

ZENDEFNODE(TreeLinearFit, {
    {
        {"float", "in", "0"},
        {"float", "inMin", "0"},
        {"float", "inMax", "1"},
        {"float", "outMin", "0"},
        {"float", "outMax", "1"},
    },
    {
        {"float", "out"},
    },
    {
        {"bool", "clamped", "0"},
    },
    {"tree"},
});


}
