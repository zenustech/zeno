#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


struct If : dop::Node {
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
        auto cond = value_cast<bool>(resolve(inputs.at(0), visited));
        if (cond) {
            touch(inputs.at(1), tolink, visited);
        } else {
            touch(inputs.at(2), tolink, visited);
        }
    }

    void apply() override {}
};

ZENO_DOP_DEFCLASS(If, {{
    "misc", "only runs one of the two chain by condition",
}, {
    {"cond", "bool"},
    {"then", "any"},
    {"else", "any"},
}, {
}});


struct ForData {
    int times;
};


struct ForBegin : dop::Node {
    void apply() override {
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
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
        auto fordata = value_cast<ForData>(resolve(inputs.at(0), visited));
        for (int i = 0; i < fordata.times; i++) {
            auto tmp_visited = visited;
            resolve(inputs.at(1), tmp_visited);
        }
    }

    void apply() override {}
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
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
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
    void apply() override {
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
    void apply() override {
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
    void apply() override {
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
