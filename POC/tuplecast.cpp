#include <cstdio>
#include <tuple>
#include <any>

template <class Tuple, class F, size_t ...Indices>
auto _impl_tuple_map(Tuple &&tuple, F &&f, std::index_sequence<Indices...>) {
    return std::make_tuple(f(std::get<Indices>(std::forward<Tuple>(tuple)))...);
}

template <class Tuple, class F>
auto tuple_map(Tuple &&tuple, F &&f) {
    constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;
    return _impl_tuple_map(std::forward<Tuple>(tuple), std::forward<F>(f),
                  std::make_index_sequence<N>{});
}

int main() {
    std::tuple<int, float> tup;
    auto res = tuple_map(tup, [=](auto &&x) {
            return 32.14 + x;
    });
    printf("%s\n", typeid(tup).name());
    printf("%s\n", typeid(res).name());
}
