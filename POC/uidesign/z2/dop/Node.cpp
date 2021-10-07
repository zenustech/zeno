#include <z2/dop/Node.h>
#include <z2/ztd/functional.h>


namespace z2::dop {


std::any Node::get_input(int idx) const {
    return getval(inputs.at(idx));
}


void Node::set_output(int idx, std::any val) {
    outputs.at(idx) = std::move(val);
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
    return std::visit(ztd::match([&] (Input_Link const &input) {
        input.node->preapply(tolink, visited);
    }, [&] (Input_Value const &) {
    }), input);
}


std::any resolve(Input const &input, std::set<Node *> &visited) {
    return std::visit(ztd::match([&] (Input_Link const &input) {
        std::vector<Node *> tolink;
        touch(input, tolink, visited);
        sortexec(tolink, visited);
        return input.node->outputs.at(input.sockid);
    }, [&] (Input_Value const &val) {
        return val.value;
    }), input);
}


std::any getval(Input const &input) {
    return std::visit(ztd::match([&] (Input_Link const &input) {
        return input.node->outputs.at(input.sockid);
    }, [&] (Input_Value const &val) {
        return val.value;
    }), input);
}


}
