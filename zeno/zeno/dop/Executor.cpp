#include <zeno/dop/Executor.h>
#include <zeno/dop/Descriptor.h>
#include <zeno/ztd/variant.h>
#include <zeno/ztd/any_ptr.h>
#include <zeno/ztd/error.h>
#include <zeno/zmt/log.h>
#include <map>


ZENO_NAMESPACE_BEGIN
namespace dop {


void Executor::sortexec(Node *root, std::vector<Node *> &tolink) {
    struct OrderInfo {
        float new_order = 0;
        int dep_order = 0;

        constexpr bool operator<(OrderInfo const &that) const {
            return new_order < that.new_order || dep_order < that.dep_order;
        }
    };
    std::map<Node *, OrderInfo> order;
    std::vector<Node *> nodes;

    auto dfs = [&] (auto &&dfs, Node *node) -> OrderInfo & {
        OrderInfo ord{node->xpos, 0};
        if (order.contains(node)) {
            return order.at(node);

        } else {
            nodes.push_back(node);

            std::vector<Node *> reqnodes;
            for (auto const &input: node->inputs) {
                if (input.node)
                    reqnodes.push_back(input.node);
            }

            for (auto *reqnode: reqnodes) {
                auto &depord = dfs(dfs, reqnode);
                if (depord.new_order >= ord.new_order) {
                    depord.new_order = ord.new_order;
                    depord.dep_order = std::min(depord.dep_order, ord.dep_order - 1);
                }
            }

            auto it = order.emplace(node, ord).first;
            return it->second;
        }
    };
    dfs(dfs, root);

    std::sort(nodes.begin(), nodes.end(), [&] (Node *p, Node *q) {
        return order.at(p) < order.at(q);
    });

    for (auto node: nodes) {
        if (!visited.contains(node)) {
            visited.insert(node);
            ZENO_LOG_INFO("* applying node [{}]", node->name);
            current_node = node;
            node->apply();
        }
    }
}


void Executor::touch(Input const &input, std::vector<Node *> &tolink) {
    if (input.node) {
        current_node = input.node;
        input.node->preapply(tolink, this);
    }
}


ztd::any_ptr Executor::resolve(Input const &input) {
    if (input.node) {
        std::vector<Node *> tolink;
        touch(input, tolink);
        sortexec(input.node, tolink);
        return input.node->outputs.at(input.sockid);
    } else {
        return input.value;
    }
}


ztd::any_ptr Executor::getval(Input const &input) {
    if (input.node) {
        return input.node->outputs.at(input.sockid);
    } else {
        return input.value;
    }
}


ztd::any_ptr Executor::evaluate(Input const &input) {
    try {
        return resolve(input);
    } catch (std::exception const &e) {
        ZENO_LOG_ERROR("[{}] {}", current_node->name, e.what());
        return {};
    }
}


}
ZENO_NAMESPACE_END
