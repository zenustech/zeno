#include <functional>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <any>

template <typename F, typename Ret, typename A, typename... Rest>
A helper(Ret (F::*)(A, Rest...));

template <class F, class Ret, class A, class... Rest>
A helper(Ret (F::*)(A, Rest...) const);

template<typename F>
struct first_argument {
    typedef decltype( helper(&F::operator()) ) type;
};

struct IMethod {
    virtual std::any invoke(std::any &&argument) const = 0;
    virtual ~IMethod() = default;

    template <class T>
    auto operator()(T &&t) {
        return invoke(std::move(t));
    }
};

template <class F, class T>
struct Method : IMethod {
    F func;

    Method(F &&func) : func(func) {}

    virtual std::any invoke(std::any &&argument) const override {
        return func(std::any_cast<T>(argument));
    }
};

template <class T, class F>
auto make_method(F &&func) {
    return Method<F, T>(std::move(func));
}

int main() {
    auto method = make_method<int>([=] (int val) -> int {
        return -val;
    });
    int ret = std::any_cast<int>(method(1));
    std::cout << ret << std::endl;
}
