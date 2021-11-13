#include <vector>
#include <iterator>
#include <concepts>


template <class T>
concept is_iterator =
requires (T t)
{
    *t;
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
concept is_differential_iterator = is_iterator<T> &&
requires (T t, T tt, std::size_t i)
{
    t - tt;
};

template <class T>
concept is_random_iterator = is_iterator<T> &&
requires (T t, T tt, std::size_t i)
{
    t - tt;
    t += i;
    t -= i;
};


template <class T>
concept is_ranged = requires (T t)
{
    t.begin() -> is_iterator;
    t.end() -> std::template same_as<decltype(t.begin())>;
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

    range(range const &) = default;
    range &operator=(range const &) = default;
    range(range &&) = default;
    range &operator=(range &&) = default;

    template <is_ranged R>
    constexpr range(R &&r) : range(r.begin(), r.end())
    {
    }

    constexpr auto begin() const
    {
        return m_begin;
    }

    constexpr auto end() const
    {
        return m_end;
    }

    constexpr auto size() const
        requires is_differential_iterator<T>
    {
        return m_end - m_begin;
    }
};

template <is_ranged R>
range(R &&r) -> range<decltype(std::declval<R>().begin())>;


template <class F>
struct transform
{
    F m_func;

    constexpr transform(F func)
        : m_func(std::move(func))
    {
    }

    template <class T>
    friend constexpr auto operator|(range<T> const &r, transform const &self)
    {
        return self.m_func(r);
    }
};


template <is_iterator T, class F>
struct map_range : range<T>
{
    F m_func;

    map_range(range<T> r, F const &func)
        : range<T>(std::move(r))
        , m_func(std::move(func))
    {
    }

    struct iterator
    {
        typename range<T>::iterator m_it;

        constexpr decltype(auto) operator*() const
        {
            return func(*m_it);
        }

        constexpr iterator &operator++()
        {
            m_it++;
            return *this;
        }

        constexpr iterator &operator--()
        {
            m_it--;
            return *this;
        }
    };
};
