#pragma once


#include <zeno/dop/Node.h>
#include <zeno/ztd/any_ptr.h>
#include <vector>
#include <set>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct Executor {
    std::set<Node *> visited;
    Node *current_node{};

    ztd::any_ptr resolve(Input const &input);
    void touch(Input const &input);
    void sortexec(Node *root);
    static ztd::any_ptr getval(Input const &input);
    ztd::any_ptr evaluate(Input const &input);
};


}
ZENO_NAMESPACE_END
