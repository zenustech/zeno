#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/ContextManaged.h>
#include <cassert>
#include <iostream>
namespace {

struct FuncBegin : zeno::INode {
    virtual void apply() override {
        set_output("FUNC", std::make_shared<zeno::ConditionObject>());
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
    {"extraArgs"},
    {"args", "FUNC"},
    {},
    {"functional"},
});


struct FuncEnd : zeno::ContextManagedNode {
    virtual void preApply() override {
        FuncBegin *fore = nullptr;
        if (auto it = inputBounds.find("FUNC"); it != inputBounds.end()) {
            auto [sn, ss] = it->second;
            fore = dynamic_cast<FuncBegin *>(graph->nodes.at(sn).get());
            if (!fore) {
                printf("FuncEnd::FUNC must be conn to FuncBegin::FUNC!\n");
                abort();
            }
            graph->applyNode(sn);
        }
        auto func = std::make_shared<zeno::FunctionObject>();
        func->func = [this, fore] (zeno::FunctionObject::DictType const &args) {
            if (fore) fore->update_arguments(args);
            push_context();
            zeno::INode::preApply();
            pop_context();
            zeno::FunctionObject::DictType rets{};
            if (requireInput("rets")) {
                auto frets = get_input<zeno::DictObject>("rets");
                rets = frets->lut;
            }
            return rets;
        };
        set_output("function", std::move(func));
    }

    virtual void apply() override {}
};

ZENDEFNODE(FuncEnd, {
    {"rets", "FUNC"},
    {"function"},
    {},
    {"functional"},
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
    {"function", "args"},
    {"rets"},
    {},
    {"functional"},
});


struct TaskObject : zeno::IObject {
    std::shared_ptr<zeno::FunctionObject> f;
    std::shared_ptr<zeno::DictObject > data;
    void run()
    {
        auto args = get_input<zeno::DictObject>("args")->lut;
        func->call(args);
    }
};

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
//     {"functional"},
// });
// struct ParallelTask : zeno::INode{
//     virtual void apply() override {
//         auto taskList = get_input<zeno::ListObject>("TaskList");

//     }
// }
}
