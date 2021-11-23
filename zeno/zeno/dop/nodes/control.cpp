#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


struct If : dop::Node {
    virtual void preapply(dop::Executor *exec) override {
        bool cond = value_cast<bool>(exec->resolve(inputs.at(0)));
        if (cond) {
            exec->touch(inputs.at(1));
        } else {
            exec->touch(inputs.at(2));
        }
    }

    virtual void apply() override {
        if (get_input(0)) {
            set_output(0, get_input(1));
        } else {
            set_output(0, get_input(2));
        }
    }
};

ZENO_DOP_DEFCLASS(If, {{
    "misc", "only runs one of the two chain by condition",
}, {
    {"cond", "bool"},
    {"then", "any"},
    {"else", "any"},
}, {
    {"result", "any"},
}});


struct ForData {
    int times;
};


struct ForBegin : dop::Node {
    virtual void apply() override {
        ForData fordata = {
            .times = value_cast<int>(get_input(0)),
        };
        set_output(0, ztd::make_any(fordata));
    }
};

ZENO_DOP_DEFCLASS(ForBegin, {{
    "misc", "repeat a chain for multiple times (begin block)",
}, {
    {"times", "int"},
}, {
    {"FOR", "ForData"},
}});


struct ForEnd : dop::Node {
    virtual void preapply(dop::Executor *exec) override {
        auto fordata = value_cast<ForData>(exec->resolve(inputs.at(0)));
        for (int i = 0; i < fordata.times; i++) {
            auto copied_visited = exec->visited;
            std::swap(copied_visited, exec->visited);
            exec->resolve(inputs.at(1));
            std::swap(copied_visited, exec->visited);
        }
    }

    virtual void apply() override {}
};

ZENO_DOP_DEFCLASS(ForEnd, {{
    "misc", "repeat a chain for multiple times (end block)",
}, {
    {"FOR", "ForData"},
    {"chain", "any"},
}, {
}});


#if 0
struct ListForeach : dop::Node {
    void preapply(std::vector<dop::Node *> &tolink, Executor *exec) override {
        auto cond = ztd::zany_cast<int>(resolve(inputs.at(0), visited));
        for (int i = 0; i < cond; i++) {
            auto tmp_visited = visited;
            resolve(inputs.at(1), tmp_visited);
        }
    }

    void apply() override {}
};

ZENO_DOP_DEFCLASS(ListForeach, {{
    "misc", "apply for each elements in list",
}, {
    {"list"},
    {"list"},
}, {
}});
#endif


struct PrintInt : dop::Node {
    virtual void apply() override {
        auto val = value_cast<int>(get_input(0));
        printf("Print %d\n", val);
    }
};

ZENO_DOP_DEFCLASS(PrintInt, {{
    "misc", "prints a integer",
}, {
    {"value", "int"},
}, {
}});


struct Route : dop::Node {
    virtual void apply() override {
        set_output(0, get_input(0));
    }
};

ZENO_DOP_DEFCLASS(Route, {{
    "misc", "always return the first input",
}, {
    {"value", "any"},
}, {
    {"value", "any"},
}});


struct ToView : dop::Node {
    virtual void apply() override {
        set_output(0, get_input(0));
    }
};

ZENO_DOP_DEFCLASS(ToView, {{
    "misc", "send object to be viewed",
}, {
    {"object", "any"},
}, {
    {"object", "any"},
}});


}
ZENO_NAMESPACE_END
