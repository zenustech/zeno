#pragma once


#include <z2/ztd/stdafx.h>


namespace z2::dop {


struct DopNode;


using DopPromise = std::function<std::any()>;


struct DopContext {
    std::set<std::string> visited;

    DopPromise promise(DopNode *node, int idx);
};


}  // namespace z2::dop
