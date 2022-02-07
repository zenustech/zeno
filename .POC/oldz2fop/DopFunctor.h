#pragma once


#include <z2/dop/DopContext.h>


namespace z2::dop {


struct DopNode;


using DopFunctor = std::function<void(DopNode *)>;


}  // namespace z2::dop
