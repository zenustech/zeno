#pragma once


#include <zeno/dop/DopContext.h>


struct DopNode;


using DopFunctor = std::function<void(DopNode *, DopContext *)>;
