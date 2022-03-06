#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/types/StringObject.h>
#include <algorithm>

namespace zeno {


struct ShaderCustomFuncObject : IObjectClone<ShaderCustomFuncObject> {
    std::string name;
    std::string code;
    std::vector<int> argTypes;
    int retType{};
};


struct ShaderCustomFunc : INode {
    virtual void apply() override {
        auto code = get_input<StringObject>("code")->get();
        auto args = get_input2<std::string>("args");
        auto rettype = get_input2<std::string>("rettype");

        auto func = std::make_shared<ShaderCustomFuncObject>();

        static const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        {
            auto tabid = std::find(std::begin(tab), std::end(tab), rettype) - std::begin(tab);
            if (tabid == std::size(tab))
                throw zeno::Exception("invalid return type name: " + rettype);
            func->retType = tabid + 1;
        }

        std::string exp;
        // is void possible? NO, use empty string for void!
        // if(!strcmp(args, "void") {
        //     func->code = "(" + args + ") {\n" + code + "\n}"
        // }
        // else{}
        if (args.empty() || args == "void") {} else {
            auto arglist = zeno::split_str(args, ',');
            for (auto const &argi: arglist) {
                auto argj = zeno::split_str(zeno::trim_string(argi), ' ');
                if (argj.size() != 2)
                    throw zeno::Exception("argument list syntax wrong: " + argi);
                auto type = zeno::trim_string(argj[0]);
                auto name = zeno::trim_string(argj[1]);

                auto tabid = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
                if (tabid == std::size(tab))
                    throw zeno::Exception("invalid argument type name: " + type);
                func->argTypes.push_back(tabid + 1);

                if (!exp.empty())
                    exp += ", ";
                exp += type + " " + name;
            }
        }
        func->code = "(" + exp + ") {\n" + code + "\n}";

        set_output("func", std::move(func));
    }
};


ZENDEFNODE(ShaderCustomFunc, {
    {
        {"string", "args", "vec3 arg1, vec3 arg2"},
        {"enum float vec2 vec3 vec4", "rettype", "vec3"},
        {"string", "code", "return arg1 + arg2;"},
    },
    {
        {"ShaderCustomFuncObject", "func"},
    },
    {},
    {"shader"},
});


struct ShaderInvokeFunc : ShaderNode {
    virtual int determineType(EmissionPass *em) override {
        auto func = get_input<ShaderCustomFuncObject>("func");
        auto args = get_input<ListObject>("args");
        if (args->arr.size() != func->argTypes.size())
            throw zeno::Exception("expect " + std::to_string(func->argTypes.size())
                                  + " arguments in call to " + func->name + ", got "
                                  + std::to_string(args->arr.size()));
        auto argTyIt = func->argTypes.begin();
        for (auto const &arg: args->get<std::shared_ptr<IObject>>()) {
            auto ourType = *argTyIt++;
            auto theirType = em->determineType(arg.get());
            if (ourType != theirType)
                throw zeno::Exception(std::to_string(argTyIt - func->argTypes.begin())
                                      + "-th argument expect " + em->typeNameOf(ourType)
                                      + ", got " + em->typeNameOf(theirType));
        }
        return func->retType;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto func = get_input<ShaderCustomFuncObject>("func");
        auto args = get_input<ListObject>("args");
        if (func->name.empty()) {
            EmissionPass::CommonFunc comm;
            comm.rettype = func->retType;
            comm.code = func->code;
            comm.argTypes = func->argTypes;
            func->name = em->addCommonFunc(std::move(comm));
        }
        std::string exp;
        for (auto const &arg: args->get<std::shared_ptr<IObject>>()) {
            if (!exp.empty())
                exp += ", ";
            exp += em->determineExpr(arg.get());
        }
        exp = func->name + "(" + exp + ")";

        em->emitCode(exp);
    }
};


ZENDEFNODE(ShaderInvokeFunc, {
    {
        {"ShaderCustomFuncObject", "func"},
        {"list", "args"},
    },
    {
        {"shader", "out"},
    },
    {},
    {"shader"},
});


}
