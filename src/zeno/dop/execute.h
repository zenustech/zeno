#pragma once


#include <zeno/dop/Node.h>
#include <zeno/ztd/memory.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::generic_ptr getval(Input const &input);
ztd::generic_ptr resolve(Input const &input);
ztd::generic_ptr resolve(Input const &input, std::set<Node *> &visited);
void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
void sortexec(Node *root, std::vector<Node *> &tolink, std::set<Node *> &visited);


}
ZENO_NAMESPACE_END
