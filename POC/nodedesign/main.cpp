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

struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;

    static std::any resolve(Input const &input, std::set<Node *> &visited);
    static void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
    static void sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited);
};


using Input = std::variant<std::any, Node *>;


struct Node {
    float xorder = 0;
    std::vector<Input> inputs;
    std::any result;

    virtual void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) {
        for (auto dep: inputs) {
            Graph::touch(dep, tolink, visited);
        }
        tolink.push_back(idx);
    }

    virtual void apply() = 0;
};


struct If : Node {
    void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) override {
        auto cond = std::any_cast<int>(Graph::resolve(nodes[idx].deps[0], visited));
        if (cond) {
            Graph::touch(node->deps[1], tolink, visited);
        } else {
            Graph::touch(node->deps[2], tolink, visited);
        }
    }

    void apply() override { throw "unreachable"; }
};


struct For : Node {
    void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) override {
        auto cond = std::any_cast<int>(Graph::resolve(node->deps[0], visited));
        for (int i = 0; i < cond; i++) {
            auto tmp_visited = visited;
            Graph::resolve(node->deps[1], tmp_visited);
        }
    }

    void apply() override { throw "unreachable"; }
};


struct Int : Node {
    int value = 32;

    void apply() override {
        Node::result = value;
    }
};


void Graph::sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited) {
    std::sort(tolink.begin(), tolink.end(), [&] (Node *i, Node *j) {
        return i->xorder < j->xorder;
    });
    for (auto node: tolink) {
        if (!visited.contains(node)) {
            visited.insert(node);
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
        touch(idx, tolink, visited);
        sortexec(tolink, visited);
        return node->result;
    }, [&] (auto const &) {
        return std::any{};
    }), input);
}

int main() {
    auto g = std::make_unique<Graph>();
    g->nodes.resize(5);
    g->nodes[0] = new Int{100, {}};
    g->nodes[1] = new Int{200, {}};
    g->nodes[2] = new For{"for", 400, {2, 0}};

    std::set<Node *> visited;
    g->resolve(g->nodes[2].get(), visited);
    return 0;
}
