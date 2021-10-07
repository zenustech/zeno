#pragma once


#include <z2/dop/Node.h>


namespace z2::dop {


std::any getval(Input const &input);
std::any resolve(Input const &input, std::set<Node *> &visited);
void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
void sortexec(Node *root, std::vector<Node *> &tolink, std::set<Node *> &visited);


inline std::any resolve(Input const &input) {
    std::set<Node *> visited;
    return resolve(input, visited);
}


}
