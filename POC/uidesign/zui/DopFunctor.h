#pragma once


#include "DopContext.h"


struct DopNode;


using DopFunctor = std::function<void(DopNode *, DopContext *)>;
