#pragma once


#include <zeno/zan/range.h>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_transform {


template <class F>
struct transform
{
    F m_f;

    constexpr transform(F f)
        : m_f(std::move(f))
    {
    }

    constexpr decltype(auto) operator()(is_ranged auto &&...rs) const
    {
        return m_f(range(std::forward<decltype(rs)>(rs))...);
    }

    friend constexpr decltype(auto) operator|(is_ranged auto &&r, transform const &self)
    {
        return self(std::forward<decltype(r)>(r));
    }
};


}
}
ZENO_NAMESPACE_END
