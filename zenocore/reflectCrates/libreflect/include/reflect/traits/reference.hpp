#pragma once

#include <cstdint>
#include <cstddef>
#include "reflect/polyfill.hpp"
#include "constant_eval.hpp"

namespace zeno
{
namespace reflect
{

    template <typename T>
    struct TRemoveReference {
        using Type = T;
    };

    template <typename T>
    struct TRemoveReference<T&> {
        using Type = T;
    };

    template <typename T>
    struct TRemoveReference<T&&> {
        using Type = T;
    };

    template <typename T>
    using TTRemoveReference = typename TRemoveReference<T>::Type;

    template <typename T>
    T&& forward(TTRemoveReference<T>& v) {
        return static_cast<T&&>(v);
    }

    template <typename T>
    T&& forward(TTRemoveReference<T>&& v) {
        return static_cast<T&&>(v);
    }

    template <typename T>
    struct TRemovePointer {
        using Type = T;
    };

    template <typename T>
    struct TRemovePointer<T*> {
        using Type = T;
    };

    template <typename T>
    using TTRemovePointer = typename TRemovePointer<T>::Type;

    template <typename T>
    struct TRemoveExtent {
        using Type = T;
    };

    template <typename T, size_t N>
    struct TRemoveExtent<T[N]> {
        using Type = T*;
    };

    template <typename T>
    struct TRemoveExtent<T[]> {
        using Type = T*;
    };

    template <typename T>
    struct TRemoveCV {
        using Type = T;
    };

    template <typename T>
    struct TRemoveCV<const T> {
        using Type = T;
    };

    template <typename T>
    struct TRemoveCV<volatile T> {
        using Type = T;
    };

    template <typename T>
    struct TRemoveCV<const volatile T> {
        using Type = T;
    };

    template <typename T>
    struct TDecay {
        using NoExtentType = typename TRemoveExtent<T>::Type;
        using NoCVType = typename TRemoveCV<NoExtentType>::Type;
        using Type = typename TRemoveCV<TTRemoveReference<NoCVType>>::Type;
    };

    template <typename T>
    using TTDecay = typename TDecay<T>::Type;

    template <typename T>
    struct TIsReference : TFalseType {};

    template <typename T>
    struct TIsReference<T&> : TTrueType {};

    template <typename T>
    struct TIsReference<T&&> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsReference = TIsReference<T>::value;

    template <typename T>
    struct TIsPointer : TFalseType {};

    template <typename T>
    struct TIsPointer<T*> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsPointer = TIsPointer<T>::value;
} // namespace reflect 
} // namespace zeno

