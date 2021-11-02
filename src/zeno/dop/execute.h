#pragma once


#include <zeno/dop/Node.h>
#include <zeno/ztd/any_ptr.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::any_ptr getval(Input const &input);
ztd::any_ptr resolve(Input const &input);
ztd::any_ptr resolve(Input const &input, std::set<Node *> &visited);
void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
void sortexec(Node *root, std::vector<Node *> &tolink, std::set<Node *> &visited);


}
ZENO_NAMESPACE_END
