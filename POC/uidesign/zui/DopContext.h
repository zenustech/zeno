#pragma once


#include "DopLazy.h"


struct DopContext {
    ztd::Vector<DopLazy> in;
    ztd::Vector<DopLazy> out;
};

using DopFunctor = std::function<void(DopContext *)>;
