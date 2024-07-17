#pragma once

#include "enable_if.hpp"
#include "reference.hpp"

namespace zeno
{
namespace reflect
{
    template <typename T>
    TTRemoveReference<T>&& move(T&& val) noexcept {
        return static_cast<TTRemoveReference<T>&&>(val);
    }

    template <typename T>
    TTEnableIf<VTIsMoveConstructible<T>, T&&> move_checked(T& val) {
        return move(val);
    }
}
}
