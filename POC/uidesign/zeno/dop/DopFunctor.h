#pragma once


#include <zeno/dop/DopContext.h>


namespace zeno::dop {


struct DopNode;


using DopFunctor = std::function<void(DopNode *, DopContext *)>;


}  // namespace zeno::dop
