#pragma once


#include <z2/ztd/stdafx.h>
#include <z2/ztd/functional.h>


namespace z2::dop {


using DopPromise = ztd::promise<std::any>;
using DopContext = std::set<std::string>;


}  // namespace z2::dop
