#pragma once

#include "reflect/polyfill.hpp"
#include <cstdint>

namespace zeno
{
namespace reflect
{
    // ==== Meta Program Constant ====
    template <typename T, T v>
    struct TIntegralConstant {
        static REFLECT_FORCE_CONSTEPXR T value = v;
        using ValueType = T;
        using Type = TIntegralConstant<T, v>;
        REFLECT_FORCE_CONSTEPXR operator ValueType() const noexcept {
            return value;
        }
    };

    using TTrueType = TIntegralConstant<bool, true>;
    using TFalseType = TIntegralConstant<bool, false>;

    template <typename... Ts>
    struct TMakeVoid {
        using Type = void;
    };

    template <typename... Ts>
    using TVoid = typename TMakeVoid<Ts...>::Type;
    // ==== Meta Program Constant ====

    /// @brief Get a object reference of given type for meta programming
    /// @tparam T object type
    /// @return A dangling rvalue reference of given type
    template <typename T>
    T&& declval() noexcept {
        // Hard coded casting to rvalue
        return static_cast<T&&>(*((T*)nullptr));
    }
}
}
