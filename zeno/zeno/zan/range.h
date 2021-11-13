#pragma once


#include <zeno/common.h>
#include <iterator>
#include <concepts>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_range {

template <class T>
concept is_iterator =
requires (T t, T tt)
{
    *t;
    t != tt;
};

template <class T>
concept is_forward_iterator = is_iterator<T> &&
requires (T t)
{
    ++t;
};

template <class T>
concept is_bidirectional_iterator = is_iterator<T> &&
requires (T t)
{
    ++t;
    --t;
};

template <class T>
concept is_random_iterator = is_iterator<T> &&
requires (T t, T tt, std::ptrdiff_t i)
{
    t - tt;
    t += i;
    t -= i;
};


template <class T>
concept is_ranged = requires (T t)
{
    t.begin();
    t.end();
};


template <is_iterator T>
struct range
{
    T m_begin;
    T m_end;

    using iterator = T;

    constexpr range(T begin, T end)
        : m_begin(std::move(begin)), m_end(std::move(end))
    {}

    template <is_ranged R>
    constexpr range(R &&r) : range(r.begin(), r.end())
    {
    }

    constexpr iterator begin() const
    {
        return m_begin;
    }

    constexpr iterator end() const
    {
        return m_end;
    }
};

template <is_ranged R>
range(R &&r) -> range<decltype(std::declval<R>().begin())>;


template <class R, class T>
concept is_range_of = std::same_as<std::remove_cvref_t<decltype(*std::declval<R>().begin())>, T>;


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


}
}
ZENO_NAMESPACE_END
