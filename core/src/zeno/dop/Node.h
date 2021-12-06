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
struct SubnetNode;
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

    SubnetNode *subnet = nullptr;
    std::string name;

    float xpos = 0;
    float ypos = 0;
    Descriptor const *desc = nullptr;

    Node();
    virtual ~Node();

    Node(Node &&) = delete;
    Node &operator=(Node &&) = delete;
    Node(Node const &) = delete;
    Node &operator=(Node const &) = delete;

    [[nodiscard]] ztd::any_ptr get_input(int idx) const;
    void set_output(int idx, ztd::any_ptr val);

    Node *setInput(int idx, ztd::any_ptr val);
    Node *linkInput(int idx, Node *node, int sockid);
    [[nodiscard]] ztd::any_ptr getOutput(int idx) const;

    virtual void preapply(Executor *exec);
    virtual void apply() = 0;
};


}
ZENO_NAMESPACE_END
