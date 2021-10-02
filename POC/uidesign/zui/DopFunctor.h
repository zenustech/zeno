#pragma once


#include "DopLazy.h"


using DopFunctor = std::function<void(ztd::Vector<DopLazy> const &, ztd::Vector<DopLazy> &)>;
