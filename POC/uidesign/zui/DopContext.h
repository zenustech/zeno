#pragma once


#include "stdafx.h"


struct DopContext {
    ztd::Vector<std::any> in;
    ztd::Vector<std::any> out;
};

using DopFunctor = std::function<void(DopContext *)>;
