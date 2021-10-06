#include <cstdio>
#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <set>
#include <any>


template <class ...Fs>
struct match : private Fs... {
    match(Fs &&...fs)
        : Fs(std::forward<Fs>(fs))... {}

    using Fs::operator()...;
};

template <class ...Fs>
match(Fs &&...) -> match<Fs...>;


struct Node;


using Input = std::variant<std::any, Node *>;


struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;

    static std::any getval(Input const &input);
    static std::any resolve(Input const &input, std::set<Node *> &visited);
    static void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
    static void sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited);
};


struct Node {
    float xorder = 0;
    std::vector<Input> inputs;
    std::any result;

    virtual void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) {
        for (auto node: inputs) {
            Graph::touch(node, tolink, visited);
        }
        tolink.push_back(this);
    }

    virtual void apply() = 0;
};


struct If : Node {
    void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) override {
        auto cond = std::any_cast<int>(Graph::resolve(inputs[0], visited));
        if (cond) {
            Graph::touch(inputs[1], tolink, visited);
        } else {
            Graph::touch(inputs[2], tolink, visited);
        }
    }

    void apply() override { throw "unreachable"; }
};


struct For : Node {
    void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) override {
        auto cond = std::any_cast<int>(Graph::resolve(inputs[0], visited));
        for (int i = 0; i < cond; i++) {
            auto tmp_visited = visited;
            Graph::resolve(inputs[1], tmp_visited);
        }
    }

    void apply() override { throw "unreachable"; }
};


struct Route : Node {
    void apply() override {
        auto val = std::any_cast<int>(Graph::getval(inputs[0]));
        Node::result = val;
    }
};


struct Print : Node {
    void apply() override {
        auto val = std::any_cast<int>(Graph::getval(inputs[0]));
        printf("Print %d\n", val);
    }
};


void Graph::sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited) {
    std::sort(tolink.begin(), tolink.end(), [&] (Node *i, Node *j) {
        return i->xorder < j->xorder;
    });
    for (auto node: tolink) {
        if (!visited.contains(node)) {
            visited.insert(node);
            node->apply();
        }
    }
}

void Graph::touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited) {
    return std::visit(match([&] (Node *node) {
        node->preapply(tolink, visited);
    }, [&] (auto const &) {
    }), input);
}

std::any Graph::resolve(Input const &input, std::set<Node *> &visited) {
    return std::visit(match([&] (Node *node) {
        std::vector<Node *> tolink;
        touch(node, tolink, visited);
        sortexec(tolink, visited);
        return node->result;
    }, [&] (std::any const &val) {
        return val;
    }), input);
}

std::any Graph::getval(Input const &input) {
    return std::visit(match([&] (Node *node) {
        return node->result;
    }, [&] (std::any const &val) {
        return val;
    }), input);
}

int main() {
    auto g = std::make_unique<Graph>();
    g->nodes.resize(5);

    g->nodes[0] = std::make_unique<Route>();
    g->nodes[0]->inputs.resize(1);
    g->nodes[0]->inputs[0] = (std::any)4;

    g->nodes[1] = std::make_unique<Print>();
    g->nodes[1]->inputs.resize(1);
    g->nodes[1]->inputs[0] = (std::any)42;

    g->nodes[2] = std::make_unique<For>();
    g->nodes[2]->inputs.resize(2);
    g->nodes[2]->inputs[0] = g->nodes[0].get();
    g->nodes[2]->inputs[1] = g->nodes[1].get();

    std::set<Node *> visited;
    g->resolve(g->nodes[2].get(), visited);
    //printf("%d\n", std::any_cast<int>(ret));
    return 0;
}
