#pragma once


#include "stdafx.h"


struct DopNode;

using DopFunctor = std::function<void(DopNode *, std::set<std::string> &)>;
