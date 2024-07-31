#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/ContextManaged.h>
#include <zeno/extra/MethodCaller.h>
namespace zeno {
namespace {

struct FuncBegin : zeno::INode {
    virtual void apply() override {
        set_output("FUNC", std::make_shared<zeno::DummyObject>());
    }

    void update_arguments(zeno::FunctionObject::DictType const &callargs) {
        auto args = std::make_shared<zeno::DictObject>();
        if (has_input("extraArgs")) {
            auto extraArgs = get_input<zeno::DictObject>("extraArgs");
            for (auto const &[key, ptr]: extraArgs->lut) {
                args->lut[key] = ptr;
            }
        }
        for (auto const &[key, ptr]: callargs) {
            args->lut[key] = ptr;
        }
        set_output("args", std::move(args));
    }
};

ZENDEFNODE(FuncBegin, {
    {{"DictObject", "extraArgs"}},
    {{"DictObject", "args"}, "FUNC"},
    {},
    {"control"},
});


struct FuncEnd : zeno::ContextManagedNode {
    virtual void preApply() override {
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        FuncBegin *fore = nullptr;
        {
            bool bExist = true;
            zeno::ParamObject paramobj = get_input_obj_param("FUNC", &bExist);
            if (paramobj.links.empty())
                throw makeError("FuncEnd::FUNC must be conn to FuncBegin::FUNC!");

            auto link = paramobj.links[0];
            fore = dynamic_cast<FuncBegin *>(spGraph->m_nodes.at(link.outNode).get());
            if (!fore) {
                throw makeError("FuncEnd::FUNC must be conn to FuncBegin::FUNC!");
            }
            spGraph->applyNode(link.outNode);
        }
        auto func = std::make_shared<zeno::FunctionObject>();
        func->func = [this, fore] (zeno::FunctionObject::DictType const &args) {
            if (fore) fore->update_arguments(args);
            

            push_context();
            zeno::INode::preApply();
            zeno::FunctionObject::DictType rets{};
            if (requireInput("rets")) {
                
                auto frets = get_input<zeno::DictObject>("rets");
                rets = frets->lut;
            }
            pop_context();
    
            return rets;
        };
        set_output("function", std::move(func));
    }

    virtual void apply() override {}
};

ZENDEFNODE(FuncEnd, {
    {
        {"DictObject", "rets", "", zeno::Socket_ReadOnly},
        {"", "FUNC", "", zeno::Socket_ReadOnly},
    },
    {"function"},
    {},
    {"control"},
});

struct FuncSimpleBegin : zeno::INode {
    virtual void apply() override {
        set_output("FUNC", std::make_shared<zeno::DummyObject>());
    }

    void update_arguments(zeno::FunctionObject::DictType const &callargs) {
        //if (callargs.size() >= 1)
            //throw makeError("FuncSimpleBegin only works with <1 argument, got {}", callargs.size());
        if (callargs.size() != 0) {
            auto arg = callargs.begin()->second;
            set_output("arg", std::move(arg));
        } else {
            set_output("arg", std::make_shared<DummyObject>());
        }
    }
};

ZENDEFNODE(FuncSimpleBegin, {
    {},
    {{"object", "arg"}, "FUNC"},
    {},
    {"control"},
});


struct FuncSimpleEnd : zeno::ContextManagedNode {
    virtual void preApply() override {
        FuncSimpleBegin *fore = nullptr;
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        {
            bool bExist = false;
            zeno::ParamObject paramobj = get_input_obj_param("FUNC", &bExist);
            if (paramobj.links.empty())
                throw makeError("FuncEnd::FUNC must be conn to FuncBegin::FUNC!");

            auto link = paramobj.links[0];
            fore = dynamic_cast<FuncSimpleBegin *>(spGraph->m_nodes.at(link.outNode).get());
            if (!fore) {
                throw makeError("FuncSimpleEnd::FUNC must be conn to FuncSimpleBegin::FUNC!");
            }
            spGraph->applyNode(link.outNode);
        }
        auto func = std::make_shared<zeno::FunctionObject>();
        func->func = [this, fore] (zeno::FunctionObject::DictType const &args) {
            if (fore) fore->update_arguments(args);
            push_context();
            zeno::INode::preApply();
            zeno::FunctionObject::DictType rets{};
            if (requireInput("ret")) {
                auto fret1 = get_input("ret");
                rets.emplace("ret", std::move(fret1));
            }
            pop_context();
    
            return rets;
        };
        set_output("function", std::move(func));
    }

    virtual void apply() override {}
};

ZENDEFNODE(FuncSimpleEnd, {
    {
        {"object", "ret", "", zeno::Socket_ReadOnly},
        {"", "FUNC", "", zeno::Socket_ReadOnly},
    },
    {"function"},
    {},
    {"control"},
});


struct FuncCall : zeno::ContextManagedNode {
    virtual void apply() override {
        if (has_input<zeno::DictObject>("function")) {
            set_output("rets", get_input("function"));
            return;
        }

        auto func = get_input<zeno::FunctionObject>("function");

        zeno::FunctionObject::DictType args{};
        if (has_input("args")) {
            args = get_input<zeno::DictObject>("args")->lut;
        }
        auto rets = std::make_shared<zeno::DictObject>();
        rets->lut = func->call(args);
        set_output("rets", std::move(rets));
    }
};

ZENDEFNODE(FuncCall, {
    {
        {"FunctionObject", "function", "", zeno::Socket_ReadOnly},
        {"FunctionObject", "args", "", zeno::Socket_ReadOnly},
    },
    {
        {"FunctionObject", "rets"},
    },
    {},
    {"control"},
});

struct FuncCallInDict : zeno::ContextManagedNode {
    virtual void apply() override {
        auto funcDict = get_input<zeno::DictObject>("funcDict");

        auto mc = !get_input2<bool>("mayNotFound")
            ? zeno::MethodCaller(funcDict, get_input2<std::string>("dictKey"))
            : zeno::MethodCaller(funcDict, get_input2<std::string>("dictKey"), {});
        if (has_input("args")) {
            mc.params = get_input<zeno::DictObject>("args")->lut;
        }
        mc.call();
        auto rets = std::make_shared<zeno::DictObject>();
        rets->lut = std::move(mc.params);
        set_output("rets", std::move(rets));
    }
};

ZENDEFNODE(FuncCallInDict, {
    {
        {"DictObject", "funcDict", "", zeno::Socket_ReadOnly},
        {"bool", "mayNotFound", "1"},
        {"string", "dictKey"},
        {"DictObject", "args", "", zeno::Socket_ReadOnly},
    },
    {
        {"DictObject", "rets"},
    },
    {},
    {"control"},
});

struct FuncSimpleCall : zeno::ContextManagedNode {
    virtual void apply() override {
        if (has_input<zeno::DictObject>("function")) {
            set_output("rets", get_input("function"));
            return;
        }

        auto func = get_input<zeno::FunctionObject>("function");

        zeno::FunctionObject::DictType args{};
        if (has_input("arg")) {
            args["arg"] = get_input("arg");
        }
        args = func->call(args);
        if (args.size() >= 1) {
            auto ret = args.begin()->second;
            set_output("ret", std::move(ret));
        } else {
            set_output("ret", std::make_shared<DummyObject>());
        }
    }
};

ZENDEFNODE(FuncSimpleCall, {
    {
        {"FunctionObject", "function", "", zeno::Socket_ReadOnly},
        {"IObject", "arg", "", zeno::Socket_ReadOnly},
    },
    {
        {"IObject", "ret"},
    },
    {},
    {"control"},
});

struct FuncSimpleCallInDict : zeno::ContextManagedNode {
    virtual void apply() override {
        auto funcDict = get_input<zeno::DictObject>("funcDict");

        auto mc = !get_input2<bool>("mayNotFound")
            ? zeno::MethodCaller(funcDict, get_input2<std::string>("dictKey"))
            : zeno::MethodCaller(funcDict, get_input2<std::string>("dictKey"), {});
        if (has_input("arg")) {
            mc.set("arg", get_input("arg"));
        }
        mc.call();
        if (mc.params.size() >= 1) {
            auto ret = mc.params.begin()->second;
            set_output("ret", std::move(ret));
            set_output2("isFound", true);
        } else {
            if (has_input("notFoundRet"))
                set_output("ret", get_input("notFoundRet"));
            else
                set_output("ret", std::make_shared<DummyObject>());
            set_output2("isFound", false);
        }
    }
};

ZENDEFNODE(FuncSimpleCallInDict, {
    {
        {"DictObject", "funcDict", "", zeno::Socket_ReadOnly},
        {"string", "dictKey"},
        {"IObject", "arg", "", zeno::Socket_ReadOnly},
        {"bool", "mayNotFound", "1"},
        {"IObject", "notFoundRet", "", zeno::Socket_ReadOnly},
    },
    {
        {"bool", "isFound"},
        {"IObject", "ret"},
    },
    {},
    {"control"},
});


// struct TaskObject : zeno::IObject {
//     std::shared_ptr<zeno::FunctionObject> f;
//     std::shared_ptr<zeno::DictObject > data;
//     void run()
//     {
//         auto args = get_input<zeno::DictObject>("args")->lut;
//         func->call(args);
//     }
// };

// struct MakeTask : zeno::INode {
//     virtual void apply() override {
//         auto f = get_input<zeno::FunctionObject>("function");
//         auto data = get_input<zeno::DictObject>("args");
//         auto task = std::make_shared<TaskObject>();
//         task->f = f;
//         task->data = data;
//         set_output("task", task);
//     } 
// };
// ZENDEFNODE(MakeTask, {
//     {"function", "args"},
//     {"task"},
//     {},
//     {"control"},
// });
// struct ParallelTask : zeno::INode{
//     virtual void apply() override {
//         auto taskList = get_input<zeno::ListObject>("TaskList");

//     }
// }
}
}
