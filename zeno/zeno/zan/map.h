
#pragma once


#include <zeno/zan/range.h>
#include <iterator>
#include <concepts>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_map {


template <class R, class F>
struct map_range
{
    R m_r;
    F m_f;

    struct iterator
    {
        typename R::iterator m_it;
        F m_f;

        constexpr decltype(auto) operator*() const
        {
            return m_f(*m_it);
        }

        constexpr iterator &operator++()
        {
            ++m_it;
            return *this;
        }

        constexpr iterator &operator--()
        {
            --m_it;
            return *this;
        }

        constexpr iterator &operator+=(std::ptrdiff_t i)
        {
            m_it += i;
            return *this;
        }

        constexpr iterator &operator-=(std::ptrdiff_t i)
        {
            m_it -= i;
            return *this;
        }

        constexpr std::ptrdiff_t operator-(iterator const &o) const
        {
            return m_it - o.m_it;
        }

        constexpr std::ptrdiff_t operator!=(iterator const &o) const
        {
            return m_it != o.m_it;
        }
    };

    constexpr iterator begin() const
    {
        return {m_r.begin(), m_f};
    }

    constexpr iterator end() const
    {
        return {m_r.end(), m_f};
    }
};

inline constexpr auto map(auto f)
{
    return transformer([=] (auto r) {
        return map_range<decltype(r), decltype(f)>{r, f};
    });
}


}
}
ZENO_NAMESPACE_END
