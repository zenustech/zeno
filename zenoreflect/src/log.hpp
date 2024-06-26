#pragma once

#include <iostream>
#include <format>
#include "args.hpp"

#define ZENO_REFLECTION_LOG_DEBUG(...) \
    if (nullptr != GLOBAL_CONTROL_FLAGS && GLOBAL_CONTROL_FLAGS->verbose) {\
        std::cout << std::format(...) << std::endl;\
    }
