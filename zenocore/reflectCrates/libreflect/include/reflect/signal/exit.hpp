#pragma once

#include <cstdint>
#include "reflect/macro.hpp"

namespace zeno
{
namespace reflect
{
    class LIBREFLECT_API IExitManager {
    public:
        static IExitManager& get();

        virtual ~IExitManager();
        virtual void graceful_exit(uint8_t exit_code) = 0;
    };
}
}
