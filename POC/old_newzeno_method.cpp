#include <functional>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <any>

/*struct Method {
    struct ArgInfo {
        std::string type;
        std::string name;
    };

    std::string name;
    std::vector<ArgInfo> arguments;
    std::vector<ArgInfo> returns;
};

struct Invoke {
    int method;
    std::vector<std::pair<int, int>> arguments;
};

std::vector<Method> methods;
std::vector<Invoke> invokes;*/

struct Type {
    virtual std::type_info const &typeinfo() const = 0;
    virtual ~Type() = default;
};

template <class T>
struct TypeT : Type {
    virtual std::type_info const &typeinfo() const override {
        return typeid(T);
    }
};

namespace details {
    template <class...>
    struct type_list {};

    template <class>
    struct type_store {};

    template <class T, class ...Ts>
    void impl_make_type_list(std::vector<std::unique_ptr<Type>> &res,
            type_list<T, Ts...>) {
        res.push_back(std::make_unique<TypeT<T>>());
        impl_make_type_list(res, type_list<Ts...>{});
    }

    void impl_make_type_list(std::vector<std::unique_ptr<Type>> &res,
            type_list<>) {
    }

    template <class ...Ts>
    void make_type_list(std::vector<std::unique_ptr<Type>> &res,
            type_store<std::tuple<Ts...>>) {
        impl_make_type_list(res, type_list<Ts...>{});
    }
}

struct Method {
    virtual std::vector<std::any> invoke(std::vector<std::any> &&args) = 0;
};

template <class Ret, class Arg, class Func>
struct Method {
    Func func;

    virtual std::vector<std::any> invoke(std::vector<std::any> &&args) {
        if (args.size() != std::tuple_size_v<Arg>) abort();
        Ret ret = func(
        details::static_tuple_for(std::tuple_size_v<Arg>, [&] (auto i) {
            return std::any_cast<std::tuple_element_t<i, Arg>>(args[i]);
        }));
        std::vector<std::any> rets(std::tuple_size_v<Ret>);
        details::static_tuple_for(std::tuple_size_v<Arg>, [&] (auto i) {
            rets[i] = std::get<i>(ret);
            return false;
        });
        return rets;
    }
};

template <class Ret, class Arg, class Func>
std::unique_ptr<Method> register_method(Func &&func) {
    std::vector<std::unique_ptr<Type>> rettypes, argtypes;
    details::make_type_list(rettypes, details::type_store<Ret>{});
    details::make_type_list(argtypes, details::type_store<Arg>{});
}

template <class Ret, class Arg>
void invoke_method(Method const &meth, ) {
    std::vector<std::unique_ptr<Type>> rettypes, argtypes;
    details::make_type_list(rettypes, details::type_store<Ret>{});
    details::make_type_list(argtypes, details::type_store<Arg>{});
}

int main() {
    auto method = register_method<std::tuple<int>, std::tuple<int, int>>
    ([=] (std::tuple<int, int> const &args) -> std::tuple<int> {
        auto const &[lhs, rhs] = args;
        return {lhs + rhs};
    });
    invoke_method(*method);
}
