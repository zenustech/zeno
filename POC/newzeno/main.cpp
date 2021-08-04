#include "Backend.h"
#include "Frontend.h"


#if 1
namespace details {
    template <class T>
    struct function_traits : function_traits<decltype(&T::operator())> {
    };

    // partial specialization for function type
    template <class R, class... Args>
    struct function_traits<R(Args...)> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    // partial specialization for function pointer
    template <class R, class... Args>
    struct function_traits<R (*)(Args...)> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    // partial specialization for std::function
    template <class R, class... Args>
    struct function_traits<std::function<R(Args...)>> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    // partial specialization for pointer-to-member-function (i.e., operator()'s)
    template <class T, class R, class... Args>
    struct function_traits<R (T::*)(Args...)> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    template <class T, class R, class... Args>
    struct function_traits<R (T::*)(Args...) const> {
        using result_type = R;
        using argument_types = std::tuple<Args...>;
    };

    template <size_t N, class T>
    using function_nth_argument_t = std::tuple_element_t<N,
          typename function_traits<T>::argument_types>;

    template <class Tuple, class List, size_t ...Indices>
    auto impl_any_list_to_tuple(List &&list, std::index_sequence<Indices...>) {
        return std::make_tuple(
                std::any_cast<std::tuple_element_t<Indices, Tuple>>
                (list[Indices])...);
    }

    template <class Tuple, class List>
    auto any_list_to_tuple(List &&list) {
        constexpr size_t N = std::tuple_size_v<Tuple>;
        return impl_any_list_to_tuple<Tuple>(
                std::forward<List>(list),
                std::make_index_sequence<N>{});
    }
}

template <class F>
auto wrap_context(F func) {
    return [=] (Context *ctx) {
        using Args = details::function_nth_argument_t<0, F>;
        auto ret = func(details::any_list_to_tuple<Args>(
                    static_cast<Context const *>(ctx)->inputs));
        ctx->outputs[0] = ret;
    };
}
#endif


int myadd(std::tuple<int, int> arguments) {
    auto [x, y] = arguments;
    auto z = x + y;
    return z;
}
static auto _def_myadd = Session::get().defineNode("myadd", wrap_context(myadd));

int makeint(std::tuple<> arguments) {
    return 21;
}
static auto _def_makeint = Session::get().defineNode("makeint", wrap_context(makeint));

int printint(std::tuple<int> arguments) {
    auto [x] = arguments;
    std::cout << "printint: " << x << std::endl;
    return 0;
}
static auto _def_printint = Session::get().defineNode("printint", wrap_context(printint));


int main() {
    Graph graph;
    graph.nodes.push_back({"makeint", {}, 1});
    graph.nodes.push_back({"myadd", {{0, 0}, {0, 0}}, 1});
    graph.nodes.push_back({"printint", {{1, 0}}, 1});

    ForwardSorter sorter(graph);
    sorter.touch(2);
    auto ir = sorter.linearize();
    for (auto const &invo: ir->invos) {
        print_invocation(invo);
    }

    auto scope = Session::get().makeScope();
    for (auto const &invo: ir->invos) {
        invo.invoke(scope.get());
    }

    return 0;
}
