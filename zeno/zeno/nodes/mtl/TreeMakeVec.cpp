#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct TreeMakeVec : TreeNode {
    int ty{};

    virtual int determineType(EmissionPass *em) override {
        auto t1 = has_input("x") ? em->determineType(get_input("x").get()) : 0;
        auto t2 = has_input("y") ? em->determineType(get_input("y").get()) : 0;
        auto t3 = has_input("z") ? em->determineType(get_input("z").get()) : 0;
        auto t4 = has_input("w") ? em->determineType(get_input("w").get()) : 0;
        ty = t1 + t2 + t3 + t4;
        if (ty > 4)
            throw zeno::Exception("TreeMakeVec expect sum of dimension to no more than 4");
        return ty;
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string exp;
        if (has_input("x")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("x").get());
        }
        if (has_input("y")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("y").get());
        }
        if (has_input("z")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("z").get());
        }
        if (has_input("w")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("w").get());
        }

        em->emitCode(em->typeNameOf(ty) + "(" + exp + ")");
    }
};


ZENDEFNODE(TreeMakeVec, {
    {
        {"float", "x", "0"},
        {"float", "y", "0"},
        {"float", "z", "0"},
        {"float", "w", "0"},
    },
    {
        {"vec4f", "out"},
    },
    {},
    {"tree"},
});


}
