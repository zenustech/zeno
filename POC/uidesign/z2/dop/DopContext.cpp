#include <z2/dop/DopContext.h>
#include <z2/dop/DopNode.h>
#include <algorithm>
#include <ranges>
#include <stack>


namespace z2::dop {


/*void DopContext::Ticket::wait() const {
    if (!ctx->visited.contains(node)) {
        ctx->visited.insert(node);
        node->_apply_func(ctx);
    }
}*/


bool DopDepsgraph::contains_node(DopNode *node) const {
    return nodeps.contains(node);
}


void DopDepsgraph::insert_node(DopNode *node, std::set<DopNode *> &&deps) {
    nodeps.emplace(node, std::move(deps));
}


void DopDepsgraph::execute() {
    std::vector<DopNode *> nodes;
    {   /* pre-sort to make x-ordering work */
        struct OrderInfo {
            float new_order = 0;
            int dep_order = 0;
            float old_order = 0;

            bool operator<(OrderInfo const &that) const {
                return new_order < that.new_order || dep_order < that.dep_order;
            }
        };

        std::map<DopNode *, OrderInfo> order;

        auto touch = [&] (auto touch, DopNode *node) -> OrderInfo & {
            OrderInfo ord{node->xpos, 0};
            if (order.contains(node)) {
                return order.at(node);
            } else {
                auto const &deps = nodeps.at(node);
                for (auto *dep: deps) {
                    auto &depord = touch(touch, dep);
                    if (depord.new_order >= ord.new_order) {
                        depord.new_order = ord.new_order;
                        depord.dep_order = std::min(depord.dep_order, ord.dep_order - 1);
                    }
                }
                auto it = order.emplace(node, ord).first;
                return it->second;
            }
        };

        for (auto const &[node, deps]: nodeps) {
            touch(touch, node);
            nodes.push_back(node);
        }

        std::sort(nodes.begin(), nodes.end(), [&] (DopNode *p, DopNode *q) {
            return order.at(p) < order.at(q);
        });
    }

#if 0
    0 = readobj;
    1 = readvdb;
    2 = getframe;
    3 = if cond=2 then=0 else=1;
    4 = visualize obj=3;

    0 = readobj;
    1 = readvdb;
    2 = getframe; <- 3
    3 = if cond=2 then=0 else=1; <- 2
    4 = visualize obj=3; <- 1

    void touch(int idx0) {
        std::stack<int> indices;
        indices.push(idx0);

        while (!indices.empty()) {
            auto idx = indices.pop();
            if (node[idx] == IF) {
                auto cond = touch(node[idx].cond);
                if (cond)
                    indices.push(node[idx].then);
                else
                    indices.push(node[idx].else);
            } else {
                indices.push(node[idx].input[i]) foreach i;
            }
        }
    }
#endif

    {   /* cihou control flow nodes */
        auto touch = [&] (auto touch, int idx0) -> std::any {
            std::stack<int> indices;
            indices.push(idx0);

            while (!indices.empty()) {
                auto idx = indices.top(); indices.pop();
                auto const &nodin = nodins.at(idx);

                if (node->kind == "if") {
                    auto cond = touch(touch, nodin[0]);
                    if (std::any_cast<int>(cond)) {
                        indices.push(nodin[1]);
                    } else {
                        indices.push(nodin[2]);
                    }

                } else {
                    for (auto const &i: nodin) {
                        indices.push(i);
                    }
                }
            }
        };
    }

    {   /* final execution */
        for (auto *node: nodes) {
            node->execute();
        }
    }
}


}
