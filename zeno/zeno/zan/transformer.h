#pragma once


#include <zeno/zan/range.h>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_transformer {


template <class F>
struct transformer
{
    F m_f;

    constexpr transformer(F f)
        : m_f(std::move(f))
    {
    }

    constexpr decltype(auto) operator()(is_ranged auto &&...rs) const
    {
        return m_f(range(std::forward<decltype(rs)>(rs))...);
    }

    friend constexpr decltype(auto) operator|(is_ranged auto &&r, transformer const &self)
    {
        return self(std::forward<decltype(r)>(r));
    }
};


template <class T = void>
static constexpr auto cast = transformer([] (auto r) {
    if constexpr (std::is_void_v<T>) {
        return range{r.begin(), r.end()};
    } else {
        return T(r.begin(), r.end());
    }
});


}
}
ZENO_NAMESPACE_END
