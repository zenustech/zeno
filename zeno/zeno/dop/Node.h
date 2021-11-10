#pragma once


#include <zeno/ztd/vector.h>
#include <zeno/ztd/any_ptr.h>
#include <string>
#include <memory>
#include <vector>
#include <set>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct Node;
struct Descriptor;
struct Executor;


struct Input {
    Node *node = nullptr;
    int sockid = 0;
    ztd::any_ptr value;
};


struct Node {
    ztd::vector<Input> inputs;
    ztd::vector<ztd::any_ptr> outputs;

    float xpos = 0;
    float ypos = 0;
    Descriptor const *desc = nullptr;

    virtual ~Node() = default;

    ztd::any_ptr get_input(int idx) const;
    void set_output(int idx, ztd::any_ptr val);

    virtual void preapply(std::vector<Node *> &tolink, Executor *exec);
    virtual void apply() = 0;
};


}
ZENO_NAMESPACE_END
