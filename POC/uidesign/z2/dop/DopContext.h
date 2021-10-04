#pragma once


#include <z2/ztd/stdafx.h>
#include <z2/ztd/functional.h>


namespace z2::dop {


struct DopNode;


using DopPromise = ztd::promise<std::any>;
//using DopContext = std::set<std::string>;


struct DopContext {
    std::vector<DopNode *> tasks;
    std::set<DopNode *> visited;

    struct Ticket {
        DopContext *ctx;
        DopNode *node;

        void wait() const;
    };

    Ticket enqueue(DopNode *node) {
        tasks.push_back(node);
        return {this, node};
    }
};


}  // namespace z2::dop
