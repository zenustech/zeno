#include <z2/dop/dop.h>


namespace z2 {
namespace {



struct If : dop::Node {
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
        auto cond = std::any_cast<int>(resolve(inputs[0], visited));
        if (cond) {
            touch(inputs[1], tolink, visited);
        } else {
            touch(inputs[2], tolink, visited);
        }
    }

    void apply() override { throw "unreachable"; }
};


struct For : dop::Node {
    void preapply(std::vector<dop::Node *> &tolink, std::set<dop::Node *> &visited) override {
        auto cond = std::any_cast<int>(resolve(inputs[0], visited));
        for (int i = 0; i < cond; i++) {
            auto tmp_visited = visited;
            resolve(inputs[1], tmp_visited);
        }
    }

    void apply() override { throw "unreachable"; }
};


struct Route : dop::Node {
    void apply() override {
        auto val = std::any_cast<int>(getval(inputs[0]));
        result = val;
    }
};


struct Print : dop::Node {
    void apply() override {
        auto val = std::any_cast<int>(getval(inputs[0]));
        printf("Print %d\n", val);
    }
};


}
}
