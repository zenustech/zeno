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
            auto copied_touched = exec->touched;
            std::swap(copied_touched, exec->touched);
            exec->resolve(inputs.at(1));
            std::swap(copied_touched, exec->touched);
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
        auto cond = ztd::zany_cast<int>(resolve(inputs.at(0), touched));
        for (int i = 0; i < cond; i++) {
            auto tmp_touched = touched;
            resolve(inputs.at(1), tmp_touched);
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


}
ZENO_NAMESPACE_END
