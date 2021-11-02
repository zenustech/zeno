#include <zeno/dop/execute.h>
#include <zeno/dop/Descriptor.h>
#include <zeno/ztd/functional.h>
#include <zeno/ztd/any_ptr.h>
#include <zeno/zmt/log.h>
#include <map>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::any_ptr resolve(Input const &input) {
    std::set<Node *> visited;
    ZENO_LOG_INFO("=== start graph ===");
    auto ret = resolve(input, visited);
    ZENO_LOG_INFO("==== end graph ====");
    return ret;
}


void sortexec(Node *root, std::vector<Node *> &tolink, std::set<Node *> &visited) {
    struct OrderInfo {
        float new_order = 0;
        int dep_order = 0;

        bool operator<(OrderInfo const &that) const {
            return new_order < that.new_order || dep_order < that.dep_order;
        }
    };
    std::map<Node *, OrderInfo> order;
    std::vector<Node *> nodes;

    auto touch = [&] (auto touch, Node *node) -> OrderInfo & {
        OrderInfo ord{node->xpos, 0};
        if (order.contains(node)) {
            return order.at(node);

        } else {
            nodes.push_back(node);
            for (auto const &input: node->inputs) {
                std::visit(ztd::match([&] (Input_Link const &input) {
                    auto &depord = touch(touch, input.node);
                    if (depord.new_order >= ord.new_order) {
                        depord.new_order = ord.new_order;
                        depord.dep_order = std::min(depord.dep_order, ord.dep_order - 1);
                    }
                }, [&] (Input_Value const &) {
                }), input);
            }
            auto it = order.emplace(node, ord).first;
            return it->second;
        }
    };

    touch(touch, root);

    std::sort(nodes.begin(), nodes.end(), [&] (Node *p, Node *q) {
        return order.at(p) < order.at(q);
    });

    for (auto node: nodes) {
        if (!visited.contains(node)) {
            visited.insert(node);
            ZENO_LOG_INFO("* applying {}@{}", node->desc->name, (void *)node);
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


ztd::any_ptr resolve(Input const &input, std::set<Node *> &visited) {
    return std::visit(ztd::match([&] (Input_Link const &input) {
        std::vector<Node *> tolink;
        touch(input, tolink, visited);
        sortexec(input.node, tolink, visited);
        return input.node->outputs.at(input.sockid);
    }, [&] (Input_Value const &val) {
        return val.value;
    }), input);
}


ztd::any_ptr getval(Input const &input) {
    return std::visit(ztd::match([&] (Input_Link const &input) {
        return input.node->outputs.at(input.sockid);
    }, [&] (Input_Value const &val) {
        return val.value;
    }), input);
}


}
ZENO_NAMESPACE_END
