#include <z2/dop/dop.h>
#include <z2/ztd/functional.h>


namespace z2::dop {


std::any Node::get_input(int idx) {
    return getval(inputs.at(idx));
}


void Node::preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) {
    for (auto node: inputs) {
        touch(node, tolink, visited);
    }
    tolink.push_back(this);
}


void sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited) {
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


void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited) {
    return std::visit(ztd::match([&] (Node *node) {
        node->preapply(tolink, visited);
    }, [&] (std::any const &) {
    }), input);
}


std::any resolve(Input const &input, std::set<Node *> &visited) {
    return std::visit(ztd::match([&] (Node *node) {
        std::vector<Node *> tolink;
        touch(node, tolink, visited);
        sortexec(tolink, visited);
        return node->result;
    }, [&] (std::any const &val) {
        return val;
    }), input);
}


std::any getval(Input const &input) {
    return std::visit(ztd::match([&] (Node *node) {
        return node->result;
    }, [&] (std::any const &val) {
        return val;
    }), input);
}


}
