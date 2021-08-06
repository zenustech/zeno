#pragma once

#include <memory>
#include <any>


namespace fp {


namespace details {
    template <size_t N, class F, class Ret, class A, class... Rest>
    A nth_argument_helper(Ret (F::*)(A, Rest...));

    template <size_t N, class F, class Ret, class A, class... Rest>
    A nth_argument_helper(Ret (F::*)(A, Rest...) const);

    template <size_t N, class F>
    struct nth_argument {
        using type = decltype(nth_argument_helper<N>(&F::operator()));
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
std::unique_ptr<Method> make_method(F &&func) {
    using T = typename details::nth_argument<0, F>::type;
    return std::make_unique<_ImplMethod<F, T>>(std::move(func));
}


}
