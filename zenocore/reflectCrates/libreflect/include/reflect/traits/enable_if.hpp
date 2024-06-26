#pragma once

#include "reflect/polyfill.hpp"
#include "constant_eval.hpp"

namespace zeno
{
namespace reflect
{
    // ==== Enable If ===
    template <bool Cond, typename T = void>
    struct TEnableIf {};
    
    template <typename T>
    struct TEnableIf<true, T> {
        using Type = T;
    };

    template <bool Cond, typename T = void>
    using TTEnableIf = typename TEnableIf<Cond, T>::Type;
    // ==== Enable If ===

    // ==== Check copy constructible ====
    template <typename T, typename = void>
    struct TIsCopyConstructible : TFalseType {};

    template <typename T>
    struct TIsCopyConstructible<T, 
        TVoid<
            decltype(T(declval<const T&>()))
        >
    > : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsCopyConstructible = TIsCopyConstructible<T>::value;
    // ==== Check copy constructible ====

    // ==== Check move constructible ====
    template <typename T, typename = void>
    struct TIsMoveConstructible : TFalseType {};

    template <typename T>
    struct TIsMoveConstructible<T, 
        TVoid<
            decltype(T(declval<T&&>()))
        >
    > : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsMoveConstructible = TIsMoveConstructible<T>::value;
    // ==== Check move constructible ====

    // ==== Check copy assignable ====
    template <typename T, typename = void>
    struct TIsCopyAssignable : TFalseType {};

    template <typename T>
    struct TIsCopyAssignable<T, TVoid<decltype(declval<T&>() = declval<const T&>())>> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsCopyAssignable = TIsCopyAssignable<T>::value;
    // ==== Check copy assignable ====

    // ==== Check move assignable ====
    template <typename T, typename = void>
    struct TIsMoveAssignable : TFalseType {};

    template <typename T>
    struct TIsMoveAssignable<T, TVoid<decltype(declval<T&>() = declval<T&&>())>> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsMoveAssignable = TIsMoveAssignable<T>::value;
    // ==== Check move assignable ====

    // ==== Condition ====
    template <bool Cond, typename TrueType = TTrueType, typename FalseType = TFalseType>
    struct TConditional {};

    template <typename TrueType, typename FalseType>
    struct TConditional<true, TrueType, FalseType> {
        using Type = FalseType;
    };

    template <typename TrueType, typename FalseType>
    struct TConditional<false, TrueType, FalseType> {
        using Type = FalseType;
    };

    template <bool Cond, typename TrueType = TTrueType, typename FalseType = TFalseType>
    using TTConditional = typename TConditional<Cond, TrueType, FalseType>::Type;
    // ==== Condition ====

    // ==== Check qualifier ====
    template <typename T>
    struct TIsConst : TFalseType {};

    template <typename T>
    struct TIsConst<const T> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsConst = TIsConst<T>::value;

    template <typename T, typename U>
    struct TIsSame : TFalseType {};

    template <typename T>
    struct TIsSame<T, T> : TTrueType {};

    template <typename T, typename U>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsSame = TIsSame<T, U>::value;
    // ==== Check qualifier ====

    // ==== Is Member Pointer ====
    template <typename T>
    struct TIsMemberPointer : TFalseType {};

    template <typename T, typename C>
    struct TIsMemberPointer<T C::*> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsMemberPointer = TIsMemberPointer<T>::value;
    // ==== Is Member Pointer ====

    template <typename>
    struct TInPlaceType {
        explicit TInPlaceType() = default;
    };

    // ==== Is Assignable ====
    template <typename T, typename U, typename = void>
    struct TIsAssignable : TFalseType {};

    template <typename T, typename U>
    struct TIsAssignable<T, U, 
        TVoid<
            decltype(declval<T>() = declval<U>())
        >
    > : TTrueType {};

    template <typename T, typename U>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsAssignable = TIsAssignable<T, U>::value;
    // ==== Is Assignable ====


} // namespace reflect 
} // namespace zeno

