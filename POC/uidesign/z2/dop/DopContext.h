#pragma once


#include <z2/ztd/stdafx.h>


namespace z2::dop {


struct DopNode;


using DopPromise = std::function<std::any()>;


struct DopContext {
    std::set<std::string> promised;
    std::set<std::string> evaluated;

    DopPromise promise(DopNode *node, int idx);
    DopPromise immediate(std::any val);
};


}  // namespace z2::dop
