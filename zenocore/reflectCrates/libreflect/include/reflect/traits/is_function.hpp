#pragma once

#include "enable_if.hpp"

namespace zeno
{
namespace reflect
{

    template <typename T>
    struct TIsFunction : TFalseType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...)> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) const> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) volatile> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) const volatile> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) &> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) const &> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) volatile &> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) const volatile &> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) &&> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) const &&> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) volatile &&> : TTrueType {};

    template <typename Ret, typename... Args>
    struct TIsFunction<Ret(Args...) const volatile &&> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsFunction = TIsFunction<T>::value;

    template<typename T>
    struct TIsMemberFunction : TFalseType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...)> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) const> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) volatile> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) const volatile> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) &> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) const &> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) volatile &> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) const volatile &> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) &&> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) const &&> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) volatile &&> : TTrueType {};

    template<typename C, typename Ret, typename... Args>
    struct TIsMemberFunction<Ret (C::*)(Args...) const volatile &&> : TTrueType {};

    template <typename T>
    LIBREFLECT_INLINE REFLECT_FORCE_CONSTEPXR bool VTIsMemberFunction = TIsMemberFunction<T>::value;
}
}
