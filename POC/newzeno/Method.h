#pragma once

#include <utility>
#include <any>


namespace zfp {


namespace details {
    template <class F, class Ret, class A, class... Rest>
    A first_argument_helper(Ret (F::*)(A, Rest...));

    template <class F, class Ret, class A, class... Rest>
    A first_argument_helper(Ret (F::*)(A, Rest...) const);

    template <class F>
    struct first_argument {
        using type = decltype(first_argument_helper(&F::operator()));
    };
}

struct Method {
    virtual std::any operator()(std::any &&argument) const = 0;
    virtual ~Method() = default;
};

template <class F, class T>
struct _ImplMethod : Method {
    F func;

    _ImplMethod(F &&func) : func(func) {}
    virtual std::any operator()(std::any &&argument) const override {
        return func(std::any_cast<T>(argument));
    }
};

template <class F>
Method make_method(F &&func) {
    using T = typename details::first_argument<F>::type;
    return _ImplMethod<F, T>(std::move(func));
}


}
