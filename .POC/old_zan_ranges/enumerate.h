#pragma once


#include <zeno/zan/transform.h>
#include <pair>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_enumerate {


template <class R>
struct enumerate_range
{
    R m_r;

    struct iterator : std::iterator_traits<typename R::iterator>
    {
        using value_type = std::pair<std::size_t,
              typename std::iterator_traits<typename R::iterator>::value_type>;

        typename R::iterator m_it;
        std::size_t m_id;

        constexpr value_type operator*() const
        {
            return value_type(m_id, *m_it);
        }

        constexpr iterator &operator++()
        {
            ++m_it;
            ++m_id;
            return *this;
        }

        constexpr iterator &operator--()
        {
            --m_it;
            --m_id;
            return *this;
        }

        constexpr iterator &operator+=(std::ptrdiff_t i)
        {
            m_it += i;
            m_id += i;
            return *this;
        }

        constexpr iterator &operator-=(std::ptrdiff_t i)
        {
            m_it -= i;
            m_id -= i;
            return *this;
        }

        constexpr std::ptrdiff_t operator-(iterator const &o) const
        {
            return m_id - o.m_id;
        }

        constexpr std::ptrdiff_t operator!=(iterator const &o) const
        {
            return m_it != o.m_it;
        }
    };

    constexpr iterator begin() const
    {
        return {.m_it = m_r.begin(), .m_id = 0};
    }

    constexpr iterator end() const
    {
        return {.m_it = m_r.end(), .m_id = m_r.end() - m_r.begin()};
    }
};

static constexpr auto enumerate = transform([] (auto r) {
    return enumerate_range<decltype(r)>{.m_r = r};
});


}
}
ZENO_NAMESPACE_END
