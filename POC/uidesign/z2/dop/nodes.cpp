#include <z2/dop/dop.h>


namespace z2 {
namespace {


struct If : dop::Node {
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
        auto cond = std::any_cast<int>(resolve(inputs.at(0), visited));
        if (cond) {
            touch(inputs.at(1), tolink, visited);
        } else {
            touch(inputs.at(2), tolink, visited);
        }
    }

    void apply() override { throw "unreachable"; }
};

Z2_DOP_DEFINE(If, {{
    "misc", "only runs one of the two chain by condition",
}, {
    {"cond"},
    {"then"},
    {"else"},
}, {
}});


struct For : dop::Node {
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
        auto cond = std::any_cast<int>(resolve(inputs.at(0), visited));
        for (int i = 0; i < cond; i++) {
            auto tmp_visited = visited;
            resolve(inputs.at(1), tmp_visited);
        }
    }

    void apply() override { throw "unreachable"; }
};

Z2_DOP_DEFINE(For, {{
    "misc", "repeat a chain for multiple times",
}, {
    {"times"},
    {"chain"},
}, {
}});


struct Route : dop::Node {
    void apply() override {
        set_output(0, get_input(0));
    }
};

Z2_DOP_DEFINE(Route, {{
    "misc", "always return the first input",
}, {
    {"value"},
}, {
    {"value"},
}});


struct PrintInt : dop::Node {
    void apply() override {
        auto val = get_input<int>(0);
        printf("Print %d\n", val);
    }
};

Z2_DOP_DEFINE(PrintInt, {{
    "misc", "prints a integer",
}, {
    {"value"},
}, {
}});


}
}
